from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 定义 API 的访问范围
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# 你的凭据 JSON 文件的路径
SERVICE_ACCOUNT_FILE = 'data/credentials.json'

# 你的 Google Sheet 的 ID
# 你可以从表格的 URL 中找到它: https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
SPREADSHEET_ID = '1eQZytLaD4sMgQEQGiySWKI4ysWmd9gmRrqhIIScfz7U'

# 你想要获取的数据范围，使用 A1 表示法
# 例如 'Sheet1!A1:B10' 表示获取名为 Sheet1 的工作表中 A1 到 B10 的单元格
RANGE_NAME = 'Sheet1!A:C'

def main():
    print("--- 脚本开始运行 ---")
    creds = None
    try:
        creds = Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        print("1. 凭据加载成功。")

    except Exception as e:
        print(f"错误：无法加载凭据文件 '{SERVICE_ACCOUNT_FILE}'。请检查文件名和路径。", file=sys.stderr)
        print(f"详细错误: {e}", file=sys.stderr)
        return # 提前退出

    try:
        print("2. 正在构建 Google Sheets API 服务...")
        service = build('sheets', 'v4', credentials=creds)
        print("3. 服务构建成功。")

        sheet = service.spreadsheets()
        print(f"4. 正在从表格 '{SPREADSHEET_ID}' 的范围 '{RANGE_NAME}' 获取数据...")
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                    range=RANGE_NAME).execute()
        print("5. API 请求成功。")

        values = result.get('values', [])

        if not values:
            print("\n结果: 未找到任何数据。请检查您的工作表名称和范围是否正确，以及该范围内是否有数据。")
            return

        print("\n--- 获取到的数据 ---")
        for row in values:
            print(row)
        print("--- 数据结束 ---")

    except HttpError as err:
        print(f"\n错误：发生 HTTP 错误。这通常是权限问题。", file=sys.stderr)
        print(f"状态码: {err.resp.status}, 原因: {err.reason}", file=sys.stderr)
        print("请重点检查：\n1. Google Sheets API是否已在Cloud Console中启用。\n2. 您的表格是否已共享给 credentials.json 中的 client_email。", file=sys.stderr)
    except Exception as e:
        print(f"\n发生未知错误: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()