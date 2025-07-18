from supabase import create_client, Client
import os

from dotenv import load_dotenv
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Use service key for server-side

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)