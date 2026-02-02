import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_FILE = 'token.json'

if not os.path.exists(TOKEN_FILE):
    print("ไม่พบไฟล์ token.json")
else:
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE)
        service = build('drive', 'v3', credentials=creds)
        user_info = service.about().get(fields="user").execute()
        email = user_info['user']['emailAddress']
        name = user_info['user']['displayName']
        print(f"✅ บัญชีปัจจุบันคือ: {email} ({name})")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
