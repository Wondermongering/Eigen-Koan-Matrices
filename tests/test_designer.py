from fastapi.testclient import TestClient
# Import the app instance from your FastAPI application file
import ekm_designer # To access app and to allow monkeypatching its matrix

# Create a single TestClient instance for all tests in this module
client = TestClient(ekm_designer.app)

# Helper function to reset the global matrix state in ekm_designer
def reset_global_matrix_state_in_module(size=4):
    """Resets the global 'matrix' instance within the 'ekm_designer' module."""
    # This directly modifies the module's global variable.
    ekm_designer.matrix = ekm_designer.create_random_ekm(size)

def test_get_designer_matrix():
    reset_global_matrix_state_in_module(size=4) # Ensure a known starting point

    response = client.get("/api/designer_matrix")
    assert response.status_code == 200
    data = response.json()
    assert 'size' in data
    assert 'cells' in data
    assert data['size'] == 4 # Default size used in reset and in ekm_designer

def test_swap_tasks_endpoint():
    # Use a specific size for this test to make assertions predictable
    reset_global_matrix_state_in_module(size=2)

    # Get initial state
    response_initial = client.get("/api/designer_matrix")
    assert response_initial.status_code == 200
    initial_data = response_initial.json()
    initial_task_rows = initial_data['task_rows']

    # Ensure we have at least 2 tasks to swap based on our reset size
    assert initial_data['size'] >= 2, "Matrix size must be at least 2 to swap tasks."

    # Perform the swap operation
    # FastAPI's TestClient for POST requests with JSON: use the `json` parameter.
    swap_payload = {'row1': 0, 'row2': 1}
    response_swap = client.post("/api/swap_tasks", json=swap_payload)
    assert response_swap.status_code == 200
    swap_json_response = response_swap.json()
    assert swap_json_response.get('status') == 'ok' # Check status from server response

    # Get updated state
    response_updated = client.get("/api/designer_matrix")
    assert response_updated.status_code == 200
    updated_data = response_updated.json()
    updated_task_rows = updated_data['task_rows']

    # Check if tasks were swapped
    # initial_task_rows[0] should now be at updated_task_rows[1]
    # initial_task_rows[1] should now be at updated_task_rows[0]
    assert updated_task_rows[0] == initial_task_rows[1]
    assert updated_task_rows[1] == initial_task_rows[0]


