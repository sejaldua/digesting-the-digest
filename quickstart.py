from __future__ import print_function
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
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

def get_all_email_labels(service):
    # print out all labels
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    if not labels:
        print('No labels found.')
    else:
        print('Labels:')
        for label in labels:
            print(label['name'], label['id'])
        
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
    resultSizeEstimate = results.get('resultSizeEstimate')
    while len(messages) % 500 == 0:
        results = service.users().messages().list(userId='me',labelIds = [label], maxResults=500, pageToken=nextPageToken).execute()
        new_messages = results.get('messages', [])
        nextPageToken = results.get('nextPageToken')
        resultSizeEstimate = results.get('resultSizeEstimate')
        messages.extend(new_messages)
    return messages

if __name__ == '__main__':
    service = get_service()
    get_all_email_labels(service)
