"""
mysql_schema_light_sampler.py

用途：
- 适用于 MySQL：自动获取数据库内所有表、表结构（字段/类型/可空/默认值）
- 对“所有字段”都从数据中抽样：为每个字段返回最多 N 个样本值（默认 5）

轻量抽样策略（重点）：
- 不使用 `SELECT DISTINCT ...`（可能触发全表/全索引扫描，且代价更高）
- 改为：`SELECT col FROM table WHERE col IS NOT NULL LIMIT row_scan_limit`
  然后在 Python 里去重，凑够 value_limit 个值即停止

依赖：
- sqlalchemy>=2.0
- pymysql（作为 MySQL 驱动）

安装：
- pip install sqlalchemy pymysql

运行示例：
- python mysql_schema_light_sampler.py --url "mysql+pymysql://user:pwd@127.0.0.1:3306/db" --out result.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from urllib.parse import quote_plus

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError


@dataclass
class ColumnInfo:
    """单个字段的信息与抽样结果。"""

    name: str
    type: str
    nullable: bool
    default: Any
    sample_values: List[Any]
    note: Optional[str] = None  # 记录抽样失败原因等


@dataclass
class TableInfo:
    """单张表的信息。"""

    schema: Optional[str]
    name: str
    columns: List[ColumnInfo]


def _redact_db_url(db_url: str) -> str:
    """简单脱敏连接串：隐藏密码部分。"""
    if "://" not in db_url:
        return db_url
    prefix, rest = db_url.split("://", 1)
    if "@" not in rest:
        return db_url
    creds, tail = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{prefix}://{user}:***@{tail}"
    return db_url


def _build_mysql_url(
    *,
    user: str,
    password: str,
    host: str,
    port: int,
    database: str,
    charset: str = "utf8mb4",
) -> str:
    """
    用组件生成 SQLAlchemy MySQL URL，并对 user/password 做 URL 编码。
    解决密码里包含 @ : / ? # & 等字符导致 URL 解析错误的问题。
    """
    u = quote_plus(user)
    p = quote_plus(password)
    h = host.strip()
    db = database.strip()
    return f"mysql+pymysql://{u}:{p}@{h}:{int(port)}/{db}?charset={charset}"


def _preflight_connect(engine: Engine) -> Tuple[Optional[str], Optional[str]]:
    """
    连接预检：返回 (server_version, current_user)，失败则抛出异常给上层处理。
    """
    with engine.connect() as conn:
        server_version = conn.execute(text("SELECT VERSION()")).scalar()
        current_user = conn.execute(text("SELECT CURRENT_USER()")).scalar()
    return (str(server_version) if server_version is not None else None,
            str(current_user) if current_user is not None else None)


def _quote_ident(engine: Engine, ident: str) -> str:
    """对表名/字段名做方言级 quote，防止关键字/特殊字符导致 SQL 报错。"""
    preparer = engine.dialect.identifier_preparer
    return preparer.quote(ident)


def _full_table_name(engine: Engine, schema: Optional[str], table: str) -> str:
    """返回带 schema 的表名（已 quote）。MySQL 中 schema 通常就是 database 名。"""
    if schema:
        return f"{_quote_ident(engine, schema)}.{_quote_ident(engine, table)}"
    return _quote_ident(engine, table)


def _light_sample_column_values(
    engine: Engine,
    schema: Optional[str],
    table: str,
    column: str,
    *,
    value_limit: int = 5,
    row_scan_limit: int = 500,
    max_text_len: int = 200,
) -> (List[Any], Optional[str]):
    """
    轻量抽样：扫描前 row_scan_limit 行（该列非 NULL），在 Python 中去重，最多返回 value_limit 个值。
    - 优点：避免 DISTINCT 带来的数据库去重成本
    - 代价：如果重复值太多，可能抽不到足够多的不同值
    """
    value_limit = int(value_limit)
    row_scan_limit = int(row_scan_limit)
    if value_limit <= 0:
        return [], "value_limit<=0，跳过抽样"
    if row_scan_limit <= 0:
        return [], "row_scan_limit<=0，跳过抽样"

    tbl = _full_table_name(engine, schema, table)
    col = _quote_ident(engine, column)

    # 注意：LIMIT 这里直接拼接整数（已转 int 且来源为参数），避免部分方言对 bind limit 的兼容性问题
    sql = text(f"SELECT {col} AS v FROM {tbl} WHERE {col} IS NOT NULL LIMIT {row_scan_limit}")

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()

        seen = set()
        out: List[Any] = []
        for (v,) in rows:
            # 对长文本做截断，避免输出 JSON 过大
            if isinstance(v, str) and len(v) > max_text_len:
                v_norm: Any = v[:max_text_len] + "...(truncated)"
            else:
                v_norm = v

            # 去重：对不可 hash 的类型做降级处理（转成 str 参与去重）
            try:
                key = v_norm
                if key in seen:
                    continue
                seen.add(key)
            except TypeError:
                key = str(v_norm)
                if key in seen:
                    continue
                seen.add(key)

            out.append(v_norm)
            if len(out) >= value_limit:
                break

        return out, None
    except SQLAlchemyError as e:
        return [], f"抽样失败: {type(e).__name__}: {e}"


def introspect_mysql(
    db_url: str,
    *,
    schema: Optional[str] = None,
    include_views: bool = False,
    value_limit: int = 5,
    row_scan_limit: int = 500,
    max_tables: Optional[int] = None,
) -> Dict[str, Any]:
    """
    入口：反射 MySQL 元数据 + 对每个字段做轻量抽样。
    """
    # pool_pre_ping：避免长连接断开导致的“假失败”
    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    dialect = engine.dialect.name.lower()
    if not dialect.startswith("mysql"):
        raise ValueError(f"当前连接方言不是 MySQL（dialect={dialect}），请确认 --url 是否为 MySQL 连接串")

    insp = inspect(engine)

    # schema 不传时，尽量使用当前 database（便于正确引用 information_schema / 反射）
    target_schema = schema
    if not target_schema:
        with engine.connect() as conn:
            target_schema = conn.execute(text("SELECT DATABASE()")).scalar()

    tables = insp.get_table_names(schema=target_schema)
    if include_views:
        tables += insp.get_view_names(schema=target_schema)

    if max_tables is not None:
        tables = tables[: int(max_tables)]

    out_tables: List[TableInfo] = []

    for tbl in tables:
        cols = insp.get_columns(tbl, schema=target_schema)
        col_infos: List[ColumnInfo] = []

        for c in cols:
            col_name = str(c.get("name"))
            col_type = str(c.get("type"))
            nullable = bool(c.get("nullable", True))
            default = c.get("default")

            sample_values, err = _light_sample_column_values(
                engine,
                target_schema,
                tbl,
                col_name,
                value_limit=value_limit,
                row_scan_limit=row_scan_limit,
            )

            col_infos.append(
                ColumnInfo(
                    name=col_name,
                    type=col_type,
                    nullable=nullable,
                    default=default,
                    sample_values=sample_values,
                    note=err,
                )
            )

        out_tables.append(TableInfo(schema=target_schema, name=tbl, columns=col_infos))

    return {
        "db_url_redacted": _redact_db_url(db_url),
        "dialect": dialect,
        "schema": target_schema,
        "table_count": len(out_tables),
        "tables": [asdict(t) for t in out_tables],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MySQL：获取所有表结构 + 轻量抽样每列最多5个值")
    # 两种配置方式：
    # 1) 直接给 --url
    # 2) 给 --host/--port/--db/--user/--password，由脚本生成 URL（会自动 URL 编码密码）
    parser.add_argument("--url", default=None, help="MySQL 连接串（SQLAlchemy URL），例如 mysql+pymysql://user:pwd@host:3306/db")
    parser.add_argument("--host", default=None, help="MySQL 主机，例如 127.0.0.1 或 10.0.0.12")
    parser.add_argument("--port", type=int, default=3306, help="MySQL 端口，默认 3306")
    parser.add_argument("--db", default=None, help="数据库名（schema/database）")
    parser.add_argument("--user", default=None, help="用户名")
    parser.add_argument("--password", default=None, help="密码（若包含 @ 等特殊字符，推荐用该参数方式）")
    parser.add_argument("--charset", default="utf8mb4", help="连接字符集，默认 utf8mb4")
    parser.add_argument("--schema", default=None, help="database 名（一般可不填，默认取 SELECT DATABASE()）")
    parser.add_argument("--include-views", action="store_true", help="是否包含视图")
    parser.add_argument("--value-limit", type=int, default=5, help="每列最多返回多少个样本值（默认5）")
    parser.add_argument("--row-scan-limit", type=int, default=500, help="每列最多扫描多少行用于抽样（默认500）")
    parser.add_argument("--max-tables", type=int, default=None, help="最多处理多少张表（用于控量）")
    parser.add_argument("--out", default=None, help="输出 JSON 文件路径（不填则打印到 stdout）")
    args = parser.parse_args()

    # 组装 URL：优先使用 --url；否则使用组件生成（自动编码）
    if args.url:
        db_url = args.url
    else:
        missing = [k for k in ("host", "db", "user", "password") if getattr(args, k) in (None, "")]
        if missing:
            raise SystemExit(
                "缺少连接配置：请提供 --url，或同时提供 "
                + " ".join([f"--{m}" for m in missing])
            )
        db_url = _build_mysql_url(
            host=args.host,
            port=args.port,
            database=args.db,
            user=args.user,
            password=args.password,
            charset=args.charset,
        )

    # 连接预检：让错误更早、更明确（尤其是 1045/网络不通/SSL 等）
    try:
        engine = create_engine(db_url, future=True, pool_pre_ping=True)
        server_version, current_user = _preflight_connect(engine)
    except OperationalError as e:
        # 常见：1045 Access denied（账号/密码/来源IP/库权限）
        msg = str(e.orig) if getattr(e, "orig", None) else str(e)
        raise SystemExit(
            "连接 MySQL 失败（OperationalError）。\n"
            f"- 连接串(脱敏): {_redact_db_url(db_url)}\n"
            f"- 详情: {msg}\n"
            "排查建议：\n"
            "- 若密码含 @ : / ? # & 等字符：请用 --host/--db/--user/--password 方式让脚本自动编码\n"
            "- 确认 MySQL 账号允许从你当前客户端 IP 登录（'user'@'host' 绑定）\n"
            "- 确认该账号对目标库有权限（GRANT）\n"
        )
    except SQLAlchemyError as e:
        raise SystemExit(
            "连接 MySQL 失败（SQLAlchemyError）。\n"
            f"- 连接串(脱敏): {_redact_db_url(db_url)}\n"
            f"- 详情: {type(e).__name__}: {e}\n"
        )

    result = introspect_mysql(
        db_url=db_url,
        schema=args.schema,
        include_views=args.include_views,
        value_limit=args.value_limit,
        row_scan_limit=args.row_scan_limit,
        max_tables=args.max_tables,
    )

    # default=str 避免 datetime/decimal 等类型 JSON 序列化失败
    # 附带预检信息，方便确认“实际连上的用户/实例”
    result["preflight"] = {"server_version": server_version, "current_user": current_user}
    s = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()


