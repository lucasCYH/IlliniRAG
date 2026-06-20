import sys
import os
import threading
import time

# Ensure we import the correct local backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db

def worker_write_read(worker_id):
    """Worker thread performing writes followed by reads."""
    print(f"[Worker {worker_id}] Starting database operations...")
    for i in range(10):
        # 1. Write Note
        note_content = f"Note from worker {worker_id} - index {i} at {time.time()}"
        db.add_note(note_content)
        
        # 2. Read Notes
        all_notes = db.get_all_notes()
        assert len(all_notes) > 0, "Saved notes should not be empty"
        
        # 3. Simulate minor processing overhead
        time.sleep(0.01)
    print(f"[Worker {worker_id}] Finished successfully.")

def run_stress_test():
    print("=== STARTING SQLITE WAL CONCURRENCY STRESS TEST ===")
    
    # Enable clean setup
    initial_notes = db.get_all_notes()
    print(f"Initial notes count: {len(initial_notes)}")
    
    # Spawn 10 concurrent threads
    threads = []
    for idx in range(10):
        t = threading.Thread(target=worker_write_read, args=(idx,))
        threads.append(t)
        t.start()
        
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    final_notes = db.get_all_notes()
    print(f"=== STRESS TEST COMPLETED SUCCESSFULLY ===")
    print(f"Final notes count: {len(final_notes)}")
    
    # Clean up test database entries
    print("Cleaning up test notes...")
    for n in db.get_all_notes():
        db.delete_note(n['id'])
    print("Cleanup finished.")
    
if __name__ == "__main__":
    run_stress_test()
