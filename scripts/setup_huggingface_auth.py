#!/usr/bin/env python3
"""
Hugging Face Authentication Setup
Helps you authenticate with Hugging Face to access gated models
"""

import os
import sys
from huggingface_hub import login, HfApi

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    print("🔐 Hugging Face Authentication Setup")
    print("=" * 40)
    print()
    print("To access Llama 3.1 8B, you need to:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' access")
    print("3. Copy the token")
    print()
    
    # Check if already logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Already logged in as: {user['name']}")
        return True
    except Exception:
        print("❌ Not logged in")
    
    print()
    print("Enter your Hugging Face token:")
    print("(You can also set it as an environment variable: export HUGGINGFACE_TOKEN=your_token)")
    
    # Try to get token from environment first
    token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if not token:
        token = input("Token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    try:
        # Login with token
        login(token=token)
        print("✅ Successfully logged in!")
        
        # Verify access
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_huggingface_auth()
    if success:
        print("\n🎉 Authentication complete! You can now download the model.")
        print("Run: python3 scripts/download_model.py")
    else:
        print("\n❌ Authentication failed. Please try again.")
        sys.exit(1)


