from apscheduler.schedulers.background import BackgroundScheduler
import time
import logging
from curation import curate_skills

# Background execution loop with APScheduler for task coordination

# Set up logging
logging.basicConfig(level=logging.INFO)
scheduler = BackgroundScheduler()

def start_scheduler():
    """Initialize and start the background scheduler."""
    try:
        # Schedule the curation process to run every 24 hours
        scheduler.add_job(curate_skills, 'interval', hours=24, misfire_grace_time=60, coalesce=True)
        
        # Start the scheduler
        scheduler.start()
        logging.info("Scheduler started successfully")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler shutdown")
        scheduler.shutdown()
    except Exception as e:
        logging.error(f"Scheduler error: {str(e)}")
        scheduler.shutdown()

if __name__ == "__main__":
    start_scheduler()