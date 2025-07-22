# Meta-Study History Maintenance Guide

**Purpose**: Keep the VSLA AI research collaboration archive up-to-date for ongoing meta-research  
**Scope**: All AI assistant interactions, git history, and research artifacts  
**Target Users**: Researchers, AI collaboration studies, future meta-analysis

---

## 🔄 **Regular Maintenance Workflow**

### **Daily Updates (During Active Development)**

#### 1. **Claude Code Session Capture**
```bash
# Run after each Claude Code session
cd /home/kenth56/vsla/docs/history
python3 extract_claude_sessions.py

# Check for new sessions
ls -la processed_sessions/ | tail -5
```

#### 2. **Git History Update**
```bash
# Capture latest git changes
cd /home/kenth56/vsla
git log --oneline --since="yesterday" > docs/history/daily_git_log.txt

# Update git analysis if major commits occurred
git log --stat --since="yesterday" >> docs/history/git_changes_log.txt
```

#### 3. **Development Status Tracking**
```bash
# Check current git status
git status --porcelain > docs/history/current_changes.txt

# Update development metrics
echo "$(date): $(git rev-list --count HEAD) commits, $(git status --porcelain | wc -l) changed files" >> docs/history/development_metrics.log
```

### **Weekly Archive Updates**

#### 1. **Session Analysis Regeneration**
```bash
# Regenerate session summary with new data
cd /home/kenth56/vsla/docs/history
python3 -c "
import json
from datetime import datetime

# Update session summary with weekly stats
summary = {
    'last_update': datetime.now().isoformat(),
    'weekly_sessions': 'count_new_sessions_here',
    'weekly_commits': 'count_new_commits_here'
}

with open('weekly_update.json', 'w') as f:
    json.dump(summary, f, indent=2)
"
```

#### 2. **Timeline Correlation Update**
```bash
# Update the complete timeline
python3 update_timeline.py  # Script to be created
```

---

## 🤖 **Multi-AI Assistant History Capture**

### **Current Status by AI Assistant**

| AI Assistant | History Capture Status | Storage Location | Capture Method |
|--------------|------------------------|------------------|----------------|
| **Claude Code** | ✅ Complete | `~/.claude/projects/` | Automated extraction |
| **Gemini** | ⚠️ Limited | Manual export required | See solutions below |
| **ChatGPT** | ⚠️ Manual | Web interface export | Manual process |
| **Other AIs** | ❌ Not captured | Various | Individual solutions |

### **Gemini History Capture Solutions**

#### **Problem**: Gemini doesn't store full conversation history locally like Claude Code

#### **Solution 1: Manual Export (Immediate)**
```bash
# Create Gemini session capture directory
mkdir -p /home/kenth56/vsla/docs/history/gemini_sessions

# Manual process for each important Gemini session:
# 1. In Gemini web interface, use browser's "Save As" → "Complete Webpage"
# 2. Save to: gemini_sessions/YYYY-MM-DD_session_description.html
# 3. Extract text content:
python3 extract_gemini_html.py gemini_sessions/YYYY-MM-DD_session.html
```

#### **Solution 2: Browser Extension (Recommended)**
Install a conversation capture extension:

```javascript
// JavaScript bookmarklet for Gemini conversation capture
// Save as bookmark and click during Gemini sessions
(function() {
    var conversation = document.querySelectorAll('[data-message-id]');
    var text = '';
    conversation.forEach(function(msg) {
        text += msg.innerText + '\n\n---\n\n';
    });
    
    var blob = new Blob([text], {type: 'text/plain'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'gemini_session_' + new Date().toISOString().split('T')[0] + '.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
})();
```

#### **Solution 3: Automated Capture Script**
```python
# gemini_auto_capture.py - Run this during Gemini sessions
import time
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By

def capture_gemini_session():
    """Automated Gemini conversation capture"""
    
    # Setup (requires selenium and chromedriver)
    driver = webdriver.Chrome()
    
    # Manual step: Navigate to Gemini and start conversation
    input("Navigate to Gemini, start conversation, then press Enter...")
    
    # Capture conversation periodically
    session_data = []
    
    while True:
        try:
            # Extract all message elements
            messages = driver.find_elements(By.CSS_SELECTOR, "[data-message-id]")
            
            current_session = []
            for msg in messages:
                current_session.append({
                    'timestamp': datetime.now().isoformat(),
                    'content': msg.text,
                    'message_id': msg.get_attribute('data-message-id')
                })
            
            # Save if conversation grew
            if len(current_session) > len(session_data):
                session_data = current_session
                
                # Save to file
                filename = f"gemini_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(f"/home/kenth56/vsla/docs/history/gemini_sessions/{filename}", 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                print(f"Captured {len(session_data)} messages")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("Capture stopped")
            break
    
    driver.quit()

if __name__ == "__main__":
    capture_gemini_session()
```

#### **Solution 4: API Integration (Future)**
```python
# gemini_api_capture.py - For when Gemini API supports conversation history
import google.generativeai as genai
import json
from datetime import datetime

def sync_gemini_conversations():
    """Sync Gemini conversations via API (when available)"""
    
    # Configure API (requires API key)
    genai.configure(api_key="YOUR_API_KEY")
    
    # Get conversation history (when API supports it)
    try:
        conversations = genai.get_conversations()  # Hypothetical API
        
        for conv in conversations:
            filename = f"gemini_api_{conv.id}_{datetime.now().strftime('%Y%m%d')}.json"
            
            with open(f"/home/kenth56/vsla/docs/history/gemini_sessions/{filename}", 'w') as f:
                json.dump(conv.to_dict(), f, indent=2)
                
        print(f"Synced {len(conversations)} Gemini conversations")
        
    except Exception as e:
        print(f"API sync not available yet: {e}")

if __name__ == "__main__":
    sync_gemini_conversations()
```

### **ChatGPT History Capture**

#### **Manual Export Process**
```bash
# 1. Go to ChatGPT Settings → Data controls → Export data
# 2. Download the ZIP file when ready
# 3. Extract to ChatGPT sessions directory
mkdir -p /home/kenth56/vsla/docs/history/chatgpt_sessions
unzip chatgpt_export.zip -d chatgpt_sessions/

# 4. Convert to standardized format
python3 convert_chatgpt_export.py chatgpt_sessions/
```

---

## 🛠 **Automation Scripts to Create**

### **1. Complete Session Extractor** (`update_all_sessions.py`)
```python
#!/usr/bin/env python3
"""
Complete AI session history updater
Captures all AI assistant conversations in standardized format
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

def update_claude_sessions():
    """Update Claude Code sessions"""
    print("Updating Claude Code sessions...")
    result = subprocess.run(['python3', 'extract_claude_sessions.py'], 
                          capture_output=True, text=True)
    return result.returncode == 0

def update_gemini_sessions():
    """Check for new Gemini sessions"""
    gemini_dir = Path("gemini_sessions")
    if not gemini_dir.exists():
        gemini_dir.mkdir()
        
    # Count existing sessions
    existing = len(list(gemini_dir.glob("*.json"))) + len(list(gemini_dir.glob("*.html")))
    print(f"Found {existing} Gemini sessions")
    
    if existing == 0:
        print("⚠️  No Gemini sessions found. Use manual capture methods.")
    
    return True

def update_git_history():
    """Update git development history"""
    print("Updating git history...")
    
    # Get latest commits
    result = subprocess.run(['git', 'log', '--pretty=format:%H|%ad|%s', '--date=iso'], 
                          capture_output=True, text=True, cwd='/home/kenth56/vsla')
    
    if result.returncode == 0:
        with open('latest_git_history.txt', 'w') as f:
            f.write(result.stdout)
        return True
    return False

def generate_unified_timeline():
    """Generate updated unified timeline"""
    print("Generating unified timeline...")
    
    # Load all session data
    claude_sessions = []
    gemini_sessions = []
    
    # Process Claude sessions
    if Path("processed_sessions/session_summary.json").exists():
        with open("processed_sessions/session_summary.json") as f:
            claude_data = json.load(f)
            claude_sessions = claude_data.get('sessions', [])
    
    # Process Gemini sessions
    gemini_dir = Path("gemini_sessions")
    for session_file in gemini_dir.glob("*.json"):
        with open(session_file) as f:
            gemini_sessions.append(json.load(f))
    
    # Create unified timeline
    timeline = {
        'last_updated': datetime.now().isoformat(),
        'claude_sessions': len(claude_sessions),
        'gemini_sessions': len(gemini_sessions),
        'total_ai_interactions': len(claude_sessions) + len(gemini_sessions)
    }
    
    with open('unified_ai_timeline.json', 'w') as f:
        json.dump(timeline, f, indent=2)
    
    print(f"✅ Timeline updated: {timeline['total_ai_interactions']} total AI sessions")
    return True

def main():
    """Main update routine"""
    print("🔄 Starting AI collaboration history update...")
    
    success_count = 0
    
    if update_claude_sessions():
        success_count += 1
        print("✅ Claude sessions updated")
    else:
        print("❌ Claude session update failed")
    
    if update_gemini_sessions():
        success_count += 1
        print("✅ Gemini sessions checked")
    else:
        print("❌ Gemini session check failed")
    
    if update_git_history():
        success_count += 1
        print("✅ Git history updated")
    else:
        print("❌ Git history update failed")
    
    if generate_unified_timeline():
        success_count += 1
        print("✅ Unified timeline generated")
    else:
        print("❌ Timeline generation failed")
    
    print(f"\n🎯 Update complete: {success_count}/4 components updated successfully")
    
    if success_count == 4:
        print("🏆 Full meta-study archive is up to date!")
    else:
        print("⚠️  Some components need attention - check logs above")

if __name__ == "__main__":
    main()
```

### **2. Gemini HTML Extractor** (`extract_gemini_html.py`)
```python
#!/usr/bin/env python3
"""
Extract Gemini conversation from HTML files
Converts saved Gemini web pages to standardized format
"""

import sys
import json
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path

def extract_gemini_conversation(html_file):
    """Extract conversation from Gemini HTML file"""
    
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Find conversation elements (update selectors based on Gemini's structure)
    messages = []
    
    # Look for message containers (these selectors may need updating)
    user_messages = soup.find_all(attrs={'data-role': 'user'})  # Hypothetical
    ai_messages = soup.find_all(attrs={'data-role': 'assistant'})  # Hypothetical
    
    # Alternative: Look for common conversation patterns
    conversation_elements = soup.find_all(['div', 'p'], string=lambda text: text and len(text.strip()) > 10)
    
    for element in conversation_elements:
        text = element.get_text().strip()
        if len(text) > 20:  # Filter out short/empty elements
            messages.append({
                'timestamp': datetime.now().isoformat(),  # Approximate
                'content': text,
                'type': 'unknown',  # Would need better detection
                'source': 'gemini_html_extract'
            })
    
    # Create output
    session_data = {
        'source_file': str(html_file),
        'extraction_date': datetime.now().isoformat(),
        'message_count': len(messages),
        'messages': messages
    }
    
    # Save as JSON
    output_file = html_file.with_suffix('.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Extracted {len(messages)} messages from {html_file}")
    print(f"📁 Saved to {output_file}")
    
    return session_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_gemini_html.py <html_file>")
        sys.exit(1)
    
    html_file = Path(sys.argv[1])
    if not html_file.exists():
        print(f"Error: File {html_file} not found")
        sys.exit(1)
    
    extract_gemini_conversation(html_file)

if __name__ == "__main__":
    main()
```

---

## 📅 **Maintenance Schedule**

### **Daily (During Active Development)**
- [ ] Run `python3 update_all_sessions.py`
- [ ] Check for new AI conversations across all platforms
- [ ] Capture any significant Gemini conversations manually

### **Weekly**
- [ ] Review and organize captured sessions
- [ ] Update development timeline analysis
- [ ] Generate weekly progress report
- [ ] Clean up temporary files

### **Monthly**
- [ ] Comprehensive archive validation
- [ ] Update research documentation
- [ ] Export data for backup
- [ ] Review and improve capture methods

### **Before Major Milestones**
- [ ] Complete session capture across all AI platforms
- [ ] Generate comprehensive timeline analysis  
- [ ] Create milestone-specific research summary
- [ ] Backup complete archive

---

## 🔧 **Setup Instructions for New Users**

### **Initial Setup**
```bash
# 1. Clone/access the VSLA repository
cd /home/kenth56/vsla/docs/history

# 2. Install required dependencies
pip3 install beautifulsoup4 selenium requests

# 3. Create necessary directories
mkdir -p {gemini_sessions,chatgpt_sessions,other_ai_sessions}

# 4. Set up automation scripts
chmod +x update_all_sessions.py extract_gemini_html.py

# 5. Test the system
python3 update_all_sessions.py
```

### **Browser Extension Setup (For Gemini)**
```javascript
// 1. Create bookmark with this JavaScript:
javascript:(function(){var conversation=document.querySelectorAll('[data-message-id]');var text='';conversation.forEach(function(msg){text+=msg.innerText+'\n\n---\n\n';});var blob=new Blob([text],{type:'text/plain'});var url=URL.createObjectURL(blob);var a=document.createElement('a');a.href=url;a.download='gemini_session_'+new Date().toISOString().split('T')[0]+'.txt';document.body.appendChild(a);a.click();document.body.removeChild(a);})();

// 2. Use bookmark during Gemini sessions to auto-download conversation
// 3. Move downloaded files to gemini_sessions/ directory
```

---

## 🚨 **Troubleshooting Common Issues**

### **Claude Code Sessions Not Updating**
```bash
# Check if Claude Code is running
ps aux | grep claude

# Verify .claude directory exists
ls -la ~/.claude/projects/

# Manual session extraction
cd /home/kenth56/vsla/docs/history
python3 extract_claude_sessions.py --verbose
```

### **Gemini Sessions Missing**
```bash
# Check browser downloads for manual captures
ls ~/Downloads/*gemini* 2>/dev/null || echo "No Gemini files found"

# Verify capture scripts are working
python3 extract_gemini_html.py --test

# Alternative: Set up browser automation
pip3 install selenium
# Download ChromeDriver from https://chromedriver.chromium.org/
```

### **Git History Gaps**
```bash
# Check for uncommitted changes
git status

# Verify git log access
git log --oneline -10

# Update git analysis
python3 -c "
import subprocess
result = subprocess.run(['git', 'log', '--stat'], capture_output=True, text=True)
print(f'Git access: {\"OK\" if result.returncode == 0 else \"FAILED\"}')
"
```

---

## 📈 **Quality Assurance Checklist**

### **Weekly Archive Validation**
- [ ] All AI assistant conversations captured
- [ ] Git history up to date with latest commits
- [ ] Timeline correlation maintains accuracy
- [ ] No data loss or corruption detected
- [ ] Backup copies created and verified

### **Monthly Research Readiness Check**
- [ ] Archive structure remains organized
- [ ] All sessions properly formatted and indexed
- [ ] Research documentation reflects current state
- [ ] Tools and scripts remain functional
- [ ] Meta-study datasets are complete and accurate

---

## 🎯 **Future Enhancements**

### **Planned Improvements**
1. **Real-time Capture**: Develop background services for automatic session capture
2. **API Integration**: Integrate with AI assistant APIs when conversation history becomes available
3. **Advanced Analytics**: Add trend analysis and collaboration pattern detection
4. **Cross-Platform Sync**: Unified interface for all AI assistant histories
5. **Research Export**: Automated generation of research-ready datasets

### **Community Contributions**
- Document improvements to capture methods
- Share scripts for additional AI assistants
- Contribute to meta-research analysis tools
- Report issues and suggest enhancements

---

**📞 Support**: For questions about maintaining this archive, refer to the complete documentation in `/home/kenth56/vsla/docs/history/README.md` or review the master archive guide in `MASTER_RESEARCH_ARCHIVE.md`.