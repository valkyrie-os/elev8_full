import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime
import cohere
import pandas as pd
import json
from fastapi import FastAPI, HTTPException

app = FastAPI()

def analyze_user_messages(messages, user_to_analyze, cohere_api_key):
    """Analyze messages from a specific user using Cohere API"""
    co = cohere.Client('VkRfL2rl1Swi3YJF8W52J6Rxz4ZJDC0aFoQk68r6')
    
    # Filter messages for specific user
    user_messages = [msg for msg in messages if msg['user'] == user_to_analyze]
    
    if not user_messages:
        print(f"No messages found for user {user_to_analyze}")
        return None
        
    # Format messages with numbers
    formatted_messages = [f"Message {i+1}: {msg['text']}" for i, msg in enumerate(user_messages)]
    combined_text = " ".join(formatted_messages)
    
    try:
        # Get key topics
        topics_response = co.generate(
            model='command',
            prompt=f"Extract 3 main topics from this text: {combined_text}. Your response should be in JSON format.",
            max_tokens=100,
            temperature=0.1
        )
        
        # Get communication style analysis
        style_response = co.generate(
            model='command',
            prompt=f"""Analyze the communication style in this text and comment on it generally (without giving specific message references). Provide a response in the following JSON format:
{{
    "analysis": "3-4 sentences analyzing the communication style (30-50 words)",
    "score": <number between 1-10>
}}
Text to analyze: {combined_text}. MAKE SURE THAT THE TEXT IS IN THE PROVIDED JSON FORMAT, THIS IS IMPORTANT""",
            max_tokens=150,
            temperature=0.1
        )
        
        analysis_results = {
            "user": user_to_analyze,
            "message_count": len(user_messages),
            "key_topics": topics_response.generations[0].text.strip(),
            "communication_style": style_response.generations[0].text.strip()
        }
        
        return analysis_results
        
    except Exception as e:
        print(f"Error during Cohere analysis: {str(e)}")
        return None

def get_all_slack_messages():
    slack_token = 'xoxb-8064306265812-8055207847542-uqLbUjIw1Zsrg8dv1IaBKBd2'
    client = WebClient(token=slack_token)
    all_messages = []

    try:
        # Get bot info and verify connection
        bot_info = client.auth_test()
        bot_id = bot_info["user_id"]
        bot_name = bot_info["user"]
        
        # Get list of all channels
        channels_response = client.conversations_list(
            types="public_channel,private_channel",
            exclude_archived=True
        )
        
        if not channels_response['ok']:
            print(f"Error getting channels: {channels_response['error']}")
            return all_messages
            
        channels = channels_response['channels']
        
        # Process each channel
        for channel in channels:
            channel_id = channel['id']
            channel_name = channel.get('name', 'Unknown')
            
            try:
                # Try to join the channel
                try:
                    join_response = client.conversations_join(channel=channel_id)
                except SlackApiError as e:
                    if e.response['error'] != 'already_in_channel':
                        continue
                
                # Get channel history
                cursor = None
                while True:
                    try:
                        history_response = client.conversations_history(
                            channel=channel_id,
                            limit=1000,
                            cursor=cursor
                        )
                        
                        if not history_response['ok']:
                            break
                            
                        messages = history_response['messages']
                        
                        # Process messages
                        for msg in messages:
                            if "user" in msg and not msg["text"].endswith("has joined the channel"):
                                try:
                                    user_info = client.users_info(user=msg["user"])
                                    username = user_info["user"]["name"]
                                    
                                    msg_data = {
                                        "channel": channel_name,
                                        "user": username,
                                        "text": msg["text"],
                                        "timestamp": datetime.fromtimestamp(float(msg["ts"])).strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                    
                                    all_messages.append(msg_data)
                                    
                                except SlackApiError:
                                    continue
                        
                        if not history_response.get('has_more', False):
                            break
                            
                        cursor = history_response['response_metadata']['next_cursor']
                        
                    except SlackApiError:
                        break
                        
            except SlackApiError:
                continue
                
    except SlackApiError as e:
        print(f"Error: {e.response['error']}")
        
    return all_messages

def analyze_user_data(messages, username):
    """
    Analyze and format messages for a specific user
    Returns dict containing analysis results
    """
    user_messages = [msg for msg in messages if msg['user'] == username]
    
    if not user_messages:
        return None
        
    df = pd.DataFrame(user_messages)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['message_length'] = df['text'].str.len()
    df['hour'] = df['timestamp'].dt.hour
    
    channel_summary = df.groupby('channel').agg({
        'text': 'count',
        'message_length': 'mean'
    }).round(2).to_dict()
    
    analysis = {
        "username": username,
        "total_messages": len(df),
        "unique_channels": len(df['channel'].unique()),
        "date_range": {
            "start": df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
            "end": df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        },
        "channel_statistics": channel_summary,
        "messages": user_messages
    }
    
    return analysis

@app.get("/analyze/{username}")
async def analyze_slack_data(username: str):
    try:
        messages = get_all_slack_messages()
        
        if len(messages) > 0:
            # Get user data analysis
            user_analysis = analyze_user_data(messages, username)
            
            if user_analysis is None:
                raise HTTPException(status_code=404, detail=f"No messages found for user {username}")
                
            return {"user_statistics": user_analysis}
            
        else:
            raise HTTPException(status_code=500, detail="No messages retrieved. Please check bot permissions and channel access.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
