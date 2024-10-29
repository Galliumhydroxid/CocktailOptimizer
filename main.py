import logging
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.discovery import Resource

import autograd.numpy as np  # Import numpy from autograd
import numpy as np
from autograd import grad    # Import grad from autograd

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = ""
ERROR_RANGE = "Kosten!Q65"
INPUT_RANGE1 = "Cocktailabend!A11:A17"
INPUT_RANGE2 = "Cocktailabend!F11:F17"
current_column=41
def build_res_block():
    global current_column
    current_column = current_column + 1
    return f"Cocktailabend!B{current_column}"

INIT_VALUES1 = np.array([
    [1.0],
    [1.5],
    [2.0],
    [1.0],
    [2.0],
    [1.0],
    [0.5],
])
INIT_VALUES2 = np.array([
    [2.0],
    [1.0],
    [1.0],
    [0.5],
    [1.0],
    [1.0],
    [0.5],
])
sheet: Resource = None
learning_rate = 0.005
num_iterations = 10
logging.basicConfig(level=logging.INFO)

import time

def rate_limiter(max_calls, period):
    """Rate limiter decorator."""
    def decorator(func):
        last_called = [0.0]  # Mutable object to keep track of time

        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < period:
                logging.info(f"Rate-limited! Sleeping for {period-elapsed} seconds...")
                time.sleep(period - elapsed)  # Sleep for the remaining time
            last_called[0] = time.time()  # Update last called time
            return func(*args, **kwargs)

        return wrapper
    return decorator


x1 = np.zeros_like(INIT_VALUES1)
x2 = np.zeros_like(INIT_VALUES2)

def main():
    global x1
    global x2
    x1 = INIT_VALUES1.copy()
    x2 = INIT_VALUES2.copy()
    put_inputs(x1, INPUT_RANGE1)
    put_inputs(x2, INPUT_RANGE2)
    # Gradient Descent loop
    for i in range(num_iterations):
        loop(i)

def put_result(res):
    body = {
      'values': str(res)
    }

    # Use the update method to update the cells
    result = sheet.values().update(
      spreadsheetId=SPREADSHEET_ID,
      range=build_res_block(),
      valueInputOption='USER_ENTERED',  # Use 'RAW' to input data as-is or 'USER_ENTERED' for Google Sheets to interpret data
      body=body
    ).execute()
    print(f"Published result: {res}")


def save_progress(input1, input2):
    state1 = get_inputs(INPUT_RANGE1)
    state2 = get_inputs(INPUT_RANGE2)
    put_inputs(input1, INPUT_RANGE1)
    put_inputs(input2, INPUT_RANGE2)
    result = get_error()
    put_inputs(state1, INPUT_RANGE1)
    put_inputs(state2, INPUT_RANGE2)
    put_result(result)




@rate_limiter(max_calls=300, period=30)
def loop(i):
    global x1
    global x2

    print(f"In iteration {i}")
    gradient1 = finite_difference_gradient(calculate_error, x1, INPUT_RANGE1, epsilon=1e-1)  # Compute the gradient at the current point
    gradient2 = finite_difference_gradient(calculate_error, x2, INPUT_RANGE2, epsilon=1e-1)

    x1 = x1 - learning_rate * gradient1  # Update the point
    x2 = x2 - learning_rate * gradient2
    put_inputs(x1, INPUT_RANGE1)
    put_inputs(x2, INPUT_RANGE2)
    #save_progress(x1, x2)

    if i % 1 == 0:  # Print the value of x and the function every 100 iterations
        logging.warning(f"Iteration {i} with inputs {x1} and {x2}")

def finite_difference_gradient(f, x, rang, epsilon=1e-6):
    grad = np.zeros_like(x)  # Initialize the gradient array with the same shape as x
    for i in range(len(x)):
        # Create copies of x to perturb
        x1 = np.copy(x)
        x2 = np.copy(x)

        # Perturb the i-th element
        x1[i] += epsilon  # Increment
        x2[i] -= epsilon  # Decrement

        y1 = f(x1, rang)
        y2 = f(x2, rang)

        # Compute the finite difference
        grad[i] = (y1 - y2) / (2 * epsilon)  # Central difference

    logging.warning(f"Got gradient: {grad}")
    return grad

def put_inputs(input: np.ndarray, range) -> np.ndarray:
    # Define the range and the new values to be written
    RANGE = range
    print(f"Weights before transformation:\n{input}")
    if type(input) is np.ndarray:
        values = input.tolist()
    else:
        values = input
    body = {
      'values': values
    }

    # Use the update method to update the cells
    result = sheet.values().update(
      spreadsheetId=SPREADSHEET_ID,
      range=RANGE,
      valueInputOption='USER_ENTERED',  # Use 'RAW' to input data as-is or 'USER_ENTERED' for Google Sheets to interpret data
      body=body
    ).execute()
    print(f"Published weights:\n{values}")
    time.sleep(1)


def get_inputs(rang) -> np.ndarray:
    result = (
        sheet.values()
        .get(spreadsheetId=SPREADSHEET_ID, range=rang)
        .execute()
    )
    return result.get("values")


def get_error():
    result = (
        sheet.values()
        .get(spreadsheetId=SPREADSHEET_ID, range=ERROR_RANGE)
        .execute()
    )
    return result.get("values")[0][0]


def calculate_error(input: np.ndarray, rang, keep_state_change=False) -> float:
    print("Saving state...")
    prev_state = get_inputs(rang)
    print(f"Got state:\n{prev_state}")
    put_inputs(input, rang)

    values = get_error()
    print(f"Got result: {values}")
    if not keep_state_change:
        print("Restoring state...")
        put_inputs(prev_state, rang)
    return float(values)


def init():
    """Shows basic usage of the Sheets API.
  Prints values from a sample spreadsheet.
  """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secrets.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        global sheet
        sheet = service.spreadsheets()
    except HttpError as err:
        print(err)


if __name__ == "__main__":
    init()
    main()
