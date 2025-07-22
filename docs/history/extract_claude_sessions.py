#!/usr/bin/env python3
"""
Claude Code Session Extractor
Extracts and processes all Claude Code chat sessions from JSONL files
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import re

def parse_jsonl_file(filepath):
    """Parse a JSONL file and extract conversation data"""
    messages = []
    session_info = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract session metadata from first message
                    if line_num == 1:
                        session_info = {
                            'session_id': data.get('sessionId'),
                            'version': data.get('version'),
                            'cwd': data.get('cwd'),
                            'start_timestamp': data.get('timestamp')
                        }
                    
                    # Extract message content
                    message_data = {
                        'uuid': data.get('uuid'),
                        'timestamp': data.get('timestamp'),
                        'type': data.get('type'),  # 'user' or 'assistant'
                        'parent_uuid': data.get('parentUuid'),
                        'content': None,
                        'tool_uses': [],
                        'tool_results': []
                    }
                    
                    # Extract message content based on type
                    if data.get('type') == 'user':
                        msg = data.get('message', {})
                        if isinstance(msg.get('content'), str):
                            message_data['content'] = msg['content']
                        elif isinstance(msg.get('content'), list):
                            # Handle tool results
                            for item in msg['content']:
                                if item.get('type') == 'tool_result':
                                    message_data['tool_results'].append({
                                        'tool_use_id': item.get('tool_use_id'),
                                        'content': item.get('content'),
                                        'is_error': item.get('is_error', False)
                                    })
                    
                    elif data.get('type') == 'assistant':
                        msg = data.get('message', {})
                        if 'content' in msg and isinstance(msg['content'], list):
                            for item in msg['content']:
                                if item.get('type') == 'text':
                                    if message_data['content'] is None:
                                        message_data['content'] = item.get('text', '')
                                    else:
                                        message_data['content'] += '\n' + item.get('text', '')
                                elif item.get('type') == 'tool_use':
                                    message_data['tool_uses'].append({
                                        'id': item.get('id'),
                                        'name': item.get('name'),
                                        'input': item.get('input', {})
                                    })
                    
                    messages.append(message_data)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None
    
    return session_info, messages

def extract_session_summary(session_info, messages):
    """Extract a summary of the session"""
    if not messages:
        return "Empty session"
    
    # Count message types
    user_messages = [m for m in messages if m['type'] == 'user']
    assistant_messages = [m for m in messages if m['type'] == 'assistant']
    
    # Extract first and last user messages for context
    first_user_msg = next((m['content'] for m in user_messages if m['content']), "No user message")
    last_user_msg = next((m['content'] for m in reversed(user_messages) if m['content']), "No user message")
    
    # Count tool uses
    tool_uses = []
    for msg in assistant_messages:
        for tool in msg.get('tool_uses', []):
            tool_uses.append(tool['name'])
    
    summary = {
        'session_id': session_info.get('session_id', 'unknown'),
        'start_timestamp': session_info.get('start_timestamp'),
        'cwd': session_info.get('cwd'),
        'message_count': len(messages),
        'user_messages': len(user_messages),
        'assistant_messages': len(assistant_messages),
        'tool_usage': dict(zip(*np.unique(tool_uses, return_counts=True))) if tool_uses else {},
        'first_user_message': first_user_msg[:200] + "..." if len(first_user_msg) > 200 else first_user_msg,
        'last_user_message': last_user_msg[:200] + "..." if len(last_user_msg) > 200 else last_user_msg
    }
    
    return summary

def convert_to_markdown(session_info, messages, filepath):
    """Convert session to markdown format"""
    session_id = session_info.get('session_id', 'unknown')
    timestamp = session_info.get('start_timestamp', 'unknown')
    cwd = session_info.get('cwd', 'unknown')
    
    # Parse timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        formatted_time = timestamp
    
    markdown = f"""# Claude Code Session: {session_id}

**Start Time**: {formatted_time}  
**Working Directory**: `{cwd}`  
**Source File**: `{filepath.name}`  
**Total Messages**: {len(messages)}

---

"""
    
    # Process messages in chronological order
    for i, msg in enumerate(messages):
        msg_time = msg.get('timestamp', '')
        try:
            dt = datetime.fromisoformat(msg_time.replace('Z', '+00:00'))
            time_str = dt.strftime('%H:%M:%S')
        except:
            time_str = msg_time
        
        if msg['type'] == 'user':
            markdown += f"## 👤 User Message ({time_str})\n\n"
            if msg['content']:
                markdown += f"{msg['content']}\n\n"
            
            # Add tool results if any
            if msg.get('tool_results'):
                markdown += "### Tool Results:\n\n"
                for result in msg['tool_results']:
                    status = "❌ Error" if result.get('is_error') else "✅ Success"
                    markdown += f"**{status}** (Tool: {result.get('tool_use_id', 'unknown')})\n"
                    markdown += f"```\n{result.get('content', '')}\n```\n\n"
        
        elif msg['type'] == 'assistant':
            markdown += f"## 🤖 Assistant Message ({time_str})\n\n"
            if msg['content']:
                markdown += f"{msg['content']}\n\n"
            
            # Add tool uses if any
            if msg.get('tool_uses'):
                markdown += "### Tool Uses:\n\n"
                for tool in msg['tool_uses']:
                    markdown += f"**{tool['name']}**\n"
                    if tool.get('input'):
                        markdown += f"```json\n{json.dumps(tool['input'], indent=2)}\n```\n\n"
        
        markdown += "---\n\n"
    
    return markdown

def main():
    """Main extraction function"""
    # Setup paths
    raw_sessions_dir = Path("/home/kenth56/vsla/docs/history/raw_sessions")
    output_dir = Path("/home/kenth56/vsla/docs/history/processed_sessions")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = []
    for root, dirs, files in os.walk(raw_sessions_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(Path(root) / file)
    
    print(f"Found {len(jsonl_files)} JSONL session files")
    
    # Process each file
    session_summaries = []
    successful_extractions = 0
    
    for filepath in jsonl_files:
        print(f"Processing: {filepath}")
        
        session_info, messages = parse_jsonl_file(filepath)
        if session_info is None:
            continue
        
        # Generate markdown
        markdown_content = convert_to_markdown(session_info, messages, filepath)
        
        # Create output filename
        session_id = session_info.get('session_id', 'unknown')
        timestamp = session_info.get('start_timestamp', '')
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y%m%d_%H%M%S')
        except:
            date_str = 'unknown'
        
        output_file = output_dir / f"{date_str}_{session_id}.md"
        
        # Write markdown file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"  ✅ Exported to: {output_file}")
            successful_extractions += 1
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")
        
        # Collect summary (simplified without numpy)
        summary = {
            'session_id': session_info.get('session_id', 'unknown'),
            'start_timestamp': session_info.get('start_timestamp'),
            'cwd': session_info.get('cwd'),
            'message_count': len(messages),
            'file_path': str(filepath),
            'output_file': str(output_file)
        }
        session_summaries.append(summary)
    
    # Create summary report
    summary_file = output_dir / "session_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'total_sessions': len(jsonl_files),
            'successful_extractions': successful_extractions,
            'sessions': session_summaries
        }, f, indent=2)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Total sessions found: {len(jsonl_files)}")
    print(f"   Successfully extracted: {successful_extractions}")
    print(f"   Output directory: {output_dir}")
    print(f"   Summary report: {summary_file}")

if __name__ == "__main__":
    main()