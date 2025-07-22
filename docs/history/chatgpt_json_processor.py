#!/usr/bin/env python3
"""
ChatGPT JSON Export Processor for VSLA Meta-Research
Converts exported ChatGPT JSON conversations to standardized format
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def process_chatgpt_json(json_file):
    """Convert ChatGPT export JSON to standardized format"""
    
    print(f"Processing {json_file.name}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ChatGPT export format can vary - handle multiple structures
    conversations = []
    
    if isinstance(data, list):
        # Direct conversation list
        conversations = data
    elif isinstance(data, dict):
        if 'conversations' in data:
            conversations = data['conversations']
        elif 'data' in data:
            conversations = data['data']
        elif 'mapping' in data:
            # Single conversation format
            conversations = [data]
        else:
            # Assume the dict itself is a conversation
            conversations = [data]
    
    processed_sessions = []
    
    for i, conv in enumerate(conversations):
        try:
            # Handle different conversation structures
            session_id = conv.get('id', conv.get('conversation_id', f"chatgpt_{datetime.now().timestamp()}_{i}"))
            
            session = {
                'platform': 'chatgpt',
                'session_id': session_id,
                'title': conv.get('title', conv.get('name', 'Untitled ChatGPT Conversation')),
                'create_time': conv.get('create_time', conv.get('created_at', conv.get('timestamp'))),
                'update_time': conv.get('update_time', conv.get('updated_at', conv.get('modified_at'))),
                'model': conv.get('model', conv.get('model_slug', 'unknown')),
                'messages': [],
                'metadata': {
                    'source_file': json_file.name,
                    'processing_date': datetime.now().isoformat(),
                    'original_structure': type(conv).__name__
                }
            }
            
            # Extract messages - handle different ChatGPT export formats
            messages_data = None
            
            if 'mapping' in conv:
                # Standard ChatGPT export format with mapping
                mapping = conv['mapping']
                message_nodes = []
                
                # Build conversation tree from mapping
                for node_id, node_data in mapping.items():
                    if node_data and 'message' in node_data and node_data['message']:
                        message_nodes.append((node_id, node_data))
                
                # Sort by creation time if available, handling None values
                message_nodes.sort(key=lambda x: x[1]['message'].get('create_time') or 0)
                
                for node_id, node_data in message_nodes:
                    message = node_data['message']
                    
                    # Skip system messages or empty messages
                    if not message or message.get('author', {}).get('role') == 'system':
                        continue
                    
                    content = message.get('content', {})
                    if content and content.get('content_type') == 'text':
                        parts = content.get('parts', [])
                        if parts and any(part.strip() for part in parts):
                            session['messages'].append({
                                'id': node_id,
                                'role': message.get('author', {}).get('role', 'unknown'),
                                'content': '\n'.join(str(part) for part in parts if part),
                                'create_time': message.get('create_time'),
                                'metadata': {
                                    'author_name': message.get('author', {}).get('name'),
                                    'status': message.get('status'),
                                    'weight': message.get('weight'),
                                    'end_turn': message.get('end_turn')
                                }
                            })
            
            elif 'messages' in conv:
                # Direct messages array
                for msg in conv['messages']:
                    if isinstance(msg, dict) and msg.get('content'):
                        session['messages'].append({
                            'id': msg.get('id', f"msg_{len(session['messages'])}"),
                            'role': msg.get('role', 'unknown'),
                            'content': msg['content'],
                            'create_time': msg.get('timestamp', msg.get('created_at')),
                            'metadata': msg.get('metadata', {})
                        })
            
            elif 'conversation' in conv:
                # Nested conversation structure
                for msg in conv['conversation']:
                    if isinstance(msg, dict) and msg.get('text'):
                        session['messages'].append({
                            'id': msg.get('id', f"msg_{len(session['messages'])}"),
                            'role': msg.get('sender', msg.get('role', 'unknown')),
                            'content': msg['text'],
                            'create_time': msg.get('timestamp'),
                            'metadata': {}
                        })
            
            # Only include conversations with actual messages
            if session['messages']:
                # Calculate session statistics
                session['statistics'] = {
                    'total_messages': len(session['messages']),
                    'user_messages': len([m for m in session['messages'] if m['role'] == 'user']),
                    'assistant_messages': len([m for m in session['messages'] if m['role'] == 'assistant']),
                    'total_characters': sum(len(m['content']) for m in session['messages']),
                    'conversation_length': 'long' if len(session['messages']) > 20 else 'medium' if len(session['messages']) > 5 else 'short'
                }
                
                processed_sessions.append(session)
                print(f"  ✅ Extracted conversation: {session['title'][:50]}... ({len(session['messages'])} messages)")
            else:
                print(f"  ⚠️ Skipped empty conversation: {session.get('title', 'Untitled')}")
                
        except Exception as e:
            print(f"  ❌ Error processing conversation {i}: {e}")
            continue
    
    return processed_sessions

def integrate_chatgpt_sessions(chatgpt_dir, output_dir):
    """Integrate all ChatGPT JSON files into standardized format"""
    
    chatgpt_path = Path(chatgpt_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not chatgpt_path.exists():
        print(f"❌ ChatGPT directory not found: {chatgpt_path}")
        print("Please place your ChatGPT JSON exports in the specified directory")
        return None
    
    json_files = list(chatgpt_path.glob('*.json'))
    if not json_files:
        print(f"❌ No JSON files found in {chatgpt_path}")
        print("Please place your ChatGPT JSON exports in this directory")
        return None
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    all_sessions = []
    processed_count = 0
    error_count = 0
    
    # Process all JSON files in directory
    for json_file in json_files:
        try:
            sessions = process_chatgpt_json(json_file)
            all_sessions.extend(sessions)
            processed_count += 1
            print(f"✅ {json_file.name}: {len(sessions)} conversations extracted")
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {json_file.name}: {e}")
    
    if not all_sessions:
        print("❌ No conversations extracted from any files")
        return None
    
    print(f"\n📊 Processing Summary:")
    print(f"   📁 Files processed: {processed_count}/{len(json_files)}")
    print(f"   💬 Total conversations: {len(all_sessions)}")
    print(f"   ❌ Errors: {error_count}")
    
    # Save individual session files
    for i, session in enumerate(all_sessions):
        # Create filename from timestamp or title
        create_time = session.get('create_time')
        if create_time:
            try:
                if isinstance(create_time, (int, float)):
                    dt = datetime.fromtimestamp(create_time)
                else:
                    dt = datetime.fromisoformat(str(create_time).replace('Z', '+00:00'))
                date_str = dt.strftime('%Y%m%d_%H%M%S')
            except:
                date_str = f"unknown_{i}"
        else:
            date_str = f"unknown_{i}"
        
        # Clean session ID for filename
        session_id = session.get('session_id', f'chatgpt_{i}')
        safe_id = ''.join(c for c in str(session_id) if c.isalnum() or c in '-_')[:8]
        
        filename = f"chatgpt_{date_str}_{safe_id}.json"
        
        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
    
    # Calculate date range - fixed to handle None values properly
    timestamps = []
    for session in all_sessions:
        create_time = session.get('create_time')
        update_time = session.get('update_time')
        if create_time is not None:
            timestamps.append(create_time)
        if update_time is not None:
            timestamps.append(update_time)
    
    date_range = {'earliest': None, 'latest': None}
    if timestamps:
        try:
            # Convert all timestamps to comparable format
            comparable_timestamps = []
            for ts in timestamps:
                if ts is not None and isinstance(ts, (int, float)):
                    comparable_timestamps.append(ts)
                elif ts is not None and isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        comparable_timestamps.append(dt.timestamp())
                    except:
                        continue
            
            if comparable_timestamps:
                date_range['earliest'] = min(comparable_timestamps)
                date_range['latest'] = max(comparable_timestamps)
        except Exception as e:
            print(f"⚠️ Could not calculate date range: {e}")
    
    # Create summary
    summary = {
        'integration_date': datetime.now().isoformat(),
        'source_files_processed': processed_count,
        'source_files_failed': error_count,
        'total_conversations': len(all_sessions),
        'total_messages': sum(s['statistics']['total_messages'] for s in all_sessions),
        'total_user_messages': sum(s['statistics']['user_messages'] for s in all_sessions),
        'total_assistant_messages': sum(s['statistics']['assistant_messages'] for s in all_sessions),
        'total_characters': sum(s['statistics']['total_characters'] for s in all_sessions),
        'date_range': date_range,
        'conversation_lengths': {
            'short': len([s for s in all_sessions if s['statistics']['conversation_length'] == 'short']),
            'medium': len([s for s in all_sessions if s['statistics']['conversation_length'] == 'medium']),
            'long': len([s for s in all_sessions if s['statistics']['conversation_length'] == 'long'])
        },
        'models_used': list(set(s.get('model', 'unknown') for s in all_sessions)),
        'sample_titles': [s['title'][:100] for s in all_sessions[:10]]
    }
    
    with open(output_path / 'chatgpt_integration_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎯 ChatGPT Integration Complete:")
    print(f"   📁 {processed_count} JSON files processed successfully")
    print(f"   💬 {len(all_sessions)} conversations extracted")
    print(f"   📝 {summary['total_messages']} total messages")
    print(f"   👤 {summary['total_user_messages']} user messages")
    print(f"   🤖 {summary['total_assistant_messages']} assistant messages")
    print(f"   📊 {summary['total_characters']:,} total characters")
    print(f"   📅 Sessions saved to {output_path}")
    print(f"   📋 Summary saved to chatgpt_integration_summary.json")
    
    return summary

def main():
    chatgpt_dir = "/home/kenth56/vsla/docs/history/chatgpt/"
    output_dir = "/home/kenth56/vsla/docs/history/processed_sessions/"
    
    print("🔄 Starting ChatGPT JSON processing...")
    print(f"   📂 Source: {chatgpt_dir}")
    print(f"   📁 Output: {output_dir}")
    print()
    
    result = integrate_chatgpt_sessions(chatgpt_dir, output_dir)
    
    if result:
        print("\n✅ ChatGPT integration successful!")
        print("Your ChatGPT conversations are now part of the VSLA meta-research archive.")
    else:
        print("\n❌ ChatGPT integration failed!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()