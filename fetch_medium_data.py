from __future__ import print_function
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import base64
import re
from tqdm import tqdm
import pandas as pd

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_service():
    """Shows basic usage of the Gmail API. """

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8000)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service
        
def get_data(service, email_type):
    # set label variable depending on email type we want to look at
    if email_type == 'Medium Daily Digest':
        label = 'Label_7938937032304597046'
    elif email_type == "MIT Download":
        label = 'Label_4668905010391853459'
    
    # Call the Gmail API
    results = service.users().messages().list(userId='me',labelIds = [label], maxResults=500).execute()
    messages = results.get('messages', [])
    nextPageToken = results.get('nextPageToken')
    # use next page token to get another batch of 500 emails until all are retrieved
    while len(messages) % 500 == 0:
        results = service.users().messages().list(userId='me',labelIds = [label], maxResults=500, pageToken=nextPageToken).execute()
        new_messages = results.get('messages', [])
        nextPageToken = results.get('nextPageToken')
        messages.extend(new_messages)
    return messages

def parse_email_digest(date, msg_body):
    """ Returns a list of article information
        title, subtitle, author, publication, minutes (reading time)"""
    
    text = re.sub(r'\(https?:\S+.*\)', '', msg_body, flags=re.MULTILINE).strip('\r')
    articles = [list(match) for match in re.findall('(.*)\r\n\r\n(.*)\r\n\r\n(.*) \r\n in (.*)\r\n*·(.*) min read', text, re.MULTILINE)]
    
    # workaround for old email Medium Daily Digest format prior to March 23, 2021
    if len(articles) == 0:
        articles = [list(match) for match in re.findall('(.*)\r\n\r\n(.*)\r\n(.*) \r\n\ in (.*)\r\n(.*) min read', text, re.MULTILINE)]
    return articles

def compile_dataset(messages):
    data = []
    for message in tqdm(messages):
        
        # Get an email by id
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        
        # Get date of email for the purpose of topic modeling over time
        for date_dict in msg['payload']['headers']:
            if date_dict['name'] == 'Date':
                date = date_dict['value']
        date = pd.to_datetime(date)

        # Get the email body and decode it from UTF-8
        content = msg['payload']['parts'][0]['body']['data']
        msg_body = base64.urlsafe_b64decode(content).decode('utf-8')
        
        # Extract article information for all articles featured in daily digest
        fetched_articles = parse_email_digest(date, msg_body)
        for articles in fetched_articles:
            data.append([date, *articles])
    
    df = pd.DataFrame(data, columns = ['Date', 'Title', 'Subtitle', 'Author', 'Publication', 'Minutes'])
    df['Minutes'] = df['Minutes'].astype(int)
    df.to_csv('raw_data/article_data_via_gmail_api.csv', index=False)

if __name__ == '__main__':
    service = get_service()
    messages = get_data(service, email_type="Medium Daily Digest")
    compile_dataset(messages)
