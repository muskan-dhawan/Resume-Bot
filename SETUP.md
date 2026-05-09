# ResumeBot - Setup Guide

## 🔒 Secure Setup (No Exposed Keys)

### Step 1: Create `.env` file
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

### Step 2: Add your API keys to `.env`
Open `.env` with a text editor and fill in:

```
TELEGRAM_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=
```

**Get your tokens:**
- **Telegram Token**: Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot`
- **Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikeys) → Create API Key

### Step 3: Run the bot
```bash
c:/ResumeBot/myenv/Scripts/python.exe bot.py
```

## ✅ Security Features

- ✅ `.env` is gitignored (never committed to git)
- ✅ API keys loaded from `.env` only, not hardcoded
- ✅ `.env.example` shows structure without secrets
- ✅ Clear error messages if keys are missing

## 🚀 Usage

1. Send **Job Description** (text) to bot
2. Send **Resume PDF** 
3. Bot returns:
   - Matching score (0-100)
   - Missing skills
   - Weak areas
   - Suggestions
   - **Improved Resume PDF** (with original design preserved)

## Commands

- `/start` - Get instructions
- `/reset` - Clear session

---

**Never commit `.env` to git!**
