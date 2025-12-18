# 查询mcp还款用户待还与还款率
_Exported on 2025/12/11 at GMT+8 18:29:20 from Cursor (2.0.77)_

---

**User**

请通过mcp工具查询2025.12.11发生还款且手机号验证通过的借款用户，截止到昨天的待还金额以及还款金额，还款率

---

**Cursor**

我来帮你查询这个数据。首先，让我搜索知识库了解相关的表结构和业务逻辑。

让我再搜索一下关于待还金额和到期日期的计算逻辑：

```sql
WITH repay_users AS (
    -- 2025.12.11发生还款且手机号验证通过的用户
    SELECT DISTINCT r.user_id
    FROM singa_bi.sgo_repayments r
    JOIN singa_bi.users u ON r.user_id = u.id
    WHERE DATE(r.repay_at) = '2025-12-11'
    AND r.repayment_status = 3
    AND u.mobile_phone_verified = 1
),
due_amount AS (
    -- 计算截止到2025-12-10的未还款金额
    SELECT 
        o.user_id,
        SUM(o.due_amount - o.repayment_amount) AS pending_due_amount
    FROM singa_bi.sgo_orders o
    JOIN repay_users ru ON o.user_id = ru.user_id
    WHERE DATE(o.expire_at) <= '2025-12-10'
    AND o.order_status NOT IN (50, 61)
    GROUP BY o.user_id
),
repayment_amount AS (
    -- 统计2025.12.11的实际还款金额
    SELECT 
        r.user_id,
        SUM(r.repayment_amount) AS repayment_amount
    FROM singa_bi.sgo_repayments r
    JOIN repay_users ru ON r.user_id = ru.user_id
    WHERE DATE(r.repay_at) = '2025-12-11'
    AND r.repayment_status = 3
    GROUP BY r.user_id
)
SELECT 
    SUM(COALESCE(d.pending_due_amount, 0)) AS 待还金额,
    SUM(COALESCE(ra.repayment_amount, 0)) AS 还款金额,
    ROUND(SUM(COALESCE(ra.repayment_amount, 0)) / (SUM(COALESCE(d.pending_due_amount, 0)) + SUM(COALESCE(ra.repayment_amount, 0)))
 * 100, 2) AS 还款率
FROM repay_users ru
LEFT JOIN due_amount d ON ru.user_id = d.user_id
LEFT JOIN repayment_amount ra ON ru.user_id = ra.user_id;

I recognize the need to validate the calculation method for repayment rate. The current approach calculates the ratio of repaid amount against total potential repayment, which requires careful verification of data sources and calculation logic. Confirming the precise definition of pending due amount and actual repayment will be crucial for accurate financial reporting.
```

现在我已经了解了表结构，让我构建并执行SQL查询：

查询完成！让我再补充一个更详细的计算，包含不同的还款率计算方式：

## 查询结果

针对 **2025年12月11日发生还款且手机号验证通过的借款用户**，查询结果如下：

| 指标 | 数值 |
|------|------|
| **用户数** | 12 人 |
| **截止昨天(12月10日)待还金额** | 209,342,295.00 |
| **今日(12月11日)还款金额** | 33,315,840.00 |
| **还款率（还款/待还）** | 15.91% |
| **还款率（还款/总额）** | 13.73% |

### 查询逻辑说明：

1. **筛选条件**：
   - 2025-12-11 发生成功还款（`repayment_status = 3`）
   - 用户手机号验证通过（`mobile_phone_verified = 1`）

2. **待还金额**：这批用户截止到2025-12-10到期订单的剩余应还金额（`due_amount - repayment_amount`）

3. **还款金额**：这批用户在2025-12-11成功还款的总金额

4. **还款率**：
   - 方式一（还款/待还）：`33,315,840 / 209,342,295 = 15.91%`
   - 方式二（还款/总额）：`33,315,840 / (209,342,295 + 33,315,840) = 13.73%`

