// Emergency Gemini Conversation Capture Script
// Use this JavaScript in your browser console during Gemini sessions
// to prevent data loss from auto-deletion

(function() {
    'use strict';
    
    console.log('🧠 Emergency Gemini Conversation Capture v1.0');
    console.log('This script captures Gemini conversations to prevent auto-deletion data loss');
    
    const captureGeminiConversation = () => {
        console.log('🔍 Scanning for conversation elements...');
        
        // Multiple selectors to handle different Gemini UI versions
        const messageSelectors = [
            '[data-message-index]',
            '[role="listitem"]', 
            '.conversation-turn',
            '[data-testid="conversation-turn"]',
            '.message-content',
            '.user-message',
            '.assistant-message',
            'div[jscontroller][data-pid]'  // Common Gemini element
        ];
        
        let allMessages = [];
        
        // Try each selector
        for (const selector of messageSelectors) {
            const elements = document.querySelectorAll(selector);
            if (elements.length > 0) {
                console.log(`✅ Found ${elements.length} elements with selector: ${selector}`);
                
                elements.forEach((element, index) => {
                    const text = element.innerText || element.textContent;
                    if (text && text.trim().length > 10) {
                        // Try to determine if this is user or assistant message
                        let role = 'unknown';
                        const elementText = text.toLowerCase();
                        const parentElement = element.parentElement;
                        
                        // Simple heuristics to detect message type
                        if (parentElement) {
                            const parentClass = parentElement.className || '';
                            const parentText = parentElement.innerText || '';
                            
                            if (parentClass.includes('user') || parentText.includes('You:')) {
                                role = 'user';
                            } else if (parentClass.includes('assistant') || parentClass.includes('gemini') || parentText.includes('Gemini:')) {
                                role = 'assistant';
                            }
                        }
                        
                        allMessages.push({
                            index: index,
                            role: role,
                            content: text.trim(),
                            timestamp: new Date().toISOString(),
                            element_info: {
                                tag: element.tagName,
                                classes: element.className,
                                selector_used: selector,
                                char_count: text.length
                            }
                        });
                    }
                });
                break; // Use first successful selector
            }
        }
        
        // If no specific selectors worked, try broader search
        if (allMessages.length === 0) {
            console.log('⚠️ Specific selectors failed, trying broader search...');
            
            const allDivs = document.querySelectorAll('div, p, span');
            const conversationTexts = [];
            
            allDivs.forEach((element, index) => {
                const text = element.innerText || element.textContent;
                if (text && text.trim().length > 50 && text.trim().length < 5000) {
                    // Skip duplicates and navigation text
                    const cleanText = text.trim();
                    if (!conversationTexts.some(existing => existing.includes(cleanText) || cleanText.includes(existing))) {
                        conversationTexts.push(cleanText);
                    }
                }
            });
            
            // Convert to message format
            conversationTexts.forEach((text, index) => {
                allMessages.push({
                    index: index,
                    role: 'unknown',
                    content: text,
                    timestamp: new Date().toISOString(),
                    element_info: {
                        method: 'broad_search',
                        char_count: text.length
                    }
                });
            });
        }
        
        // Create conversation object
        const conversation = {
            platform: 'gemini',
            capture_method: 'emergency_browser_script',
            capture_date: new Date().toISOString(),
            url: window.location.href,
            page_title: document.title,
            user_agent: navigator.userAgent,
            messages: allMessages,
            metadata: {
                total_messages: allMessages.length,
                capture_success: allMessages.length > 0,
                selectors_tried: messageSelectors,
                page_text_length: document.body.innerText.length
            }
        };
        
        console.log(`📊 Captured ${allMessages.length} messages`);
        
        if (allMessages.length === 0) {
            console.warn('⚠️ No messages captured! The page structure may have changed.');
            console.log('💡 Try manually selecting and copying the conversation text.');
            
            // Fallback: capture entire page text
            conversation.fallback_content = {
                page_text: document.body.innerText,
                html_snapshot: document.documentElement.outerHTML.slice(0, 50000) // Truncate to prevent huge files
            };
        }
        
        // Download as JSON
        const blob = new Blob([JSON.stringify(conversation, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `gemini_emergency_capture_${new Date().toISOString().slice(0,10)}_${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        console.log(`💾 Downloaded: ${link.download}`);
        return conversation;
    };
    
    // Capture immediately
    const result = captureGeminiConversation();
    
    // Offer auto-save option
    if (confirm('🔄 Enable auto-save every 2 minutes to prevent data loss during this conversation?')) {
        let autoSaveInterval = setInterval(() => {
            console.log('🔄 Auto-saving Gemini conversation...');
            captureGeminiConversation();
        }, 120000); // Every 2 minutes
        
        console.log(`✅ Auto-save enabled. Interval ID: ${autoSaveInterval}`);
        console.log(`🛑 To stop auto-save, run: clearInterval(${autoSaveInterval})`);
        
        // Store interval ID globally so user can stop it
        window.geminiAutoSaveInterval = autoSaveInterval;
    }
    
    // Add visual indicator
    const indicator = document.createElement('div');
    indicator.innerHTML = '🧠 Gemini Emergency Capture Active';
    indicator.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: #1a73e8;
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        z-index: 10000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    `;
    document.body.appendChild(indicator);
    
    // Remove indicator after 5 seconds
    setTimeout(() => {
        if (indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }, 5000);
    
    console.log('✅ Emergency capture complete!');
    console.log('📁 File saved to Downloads folder');
    console.log('🔄 Move the downloaded file to: /home/kenth56/vsla/docs/history/gemini_sessions/');
    
    return result;
    
})();

// Instructions for use:
// 1. Copy this entire script
// 2. Open browser console in Gemini (F12 → Console)
// 3. Paste the script and press Enter
// 4. The conversation will be automatically downloaded
// 5. Move the downloaded JSON file to the gemini_sessions directory