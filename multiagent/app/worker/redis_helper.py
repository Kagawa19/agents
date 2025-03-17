"""
Redis connection helper functions.
Provides robust connection retry logic for Redis.
"""

import logging
import socket
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_robust_redis_connection(max_retries: int = 5, retry_delay: int = 2):
    """
    Get a Redis connection with robust retry logic.
    Based on Celery best practices for Redis connections.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Redis connection object
    """
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    
    # Try connection with different hostnames
    hostnames = [
        'redis',           # Docker service name
        'localhost',       # Local development
        '127.0.0.1',       # Localhost IP
        socket.gethostname(),  # Current hostname
    ]
    
    # Default Redis connection options
    redis_options = {
        'port': 6379,
        'db': 0,
        'socket_timeout': 30,
        'socket_connect_timeout': 30,
        'retry_on_timeout': True,
        'health_check_interval': 30,
        'decode_responses': False
    }
    
    for attempt in range(max_retries):
        for hostname in hostnames:
            try:
                logger.info(f"Attempting Redis connection to {hostname} (attempt {attempt+1}/{max_retries})")
                
                # Create connection
                redis_conn = redis.Redis(host=hostname, **redis_options)
                
                # Verify connection with ping
                if redis_conn.ping():
                    logger.info(f"Successfully connected to Redis at {hostname}")
                    return redis_conn
                
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Failed to connect to Redis at {hostname}: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error connecting to Redis at {hostname}: {str(e)}")
        
        # Wait before retry
        if attempt < max_retries - 1:
            logger.info(f"Waiting {retry_delay}s before next Redis connection attempt")
            time.sleep(retry_delay)
    
    # All attempts failed
    logger.error(f"Failed to connect to Redis after {max_retries} attempts with hostnames: {hostnames}")
    raise ConnectionError("Could not establish connection to Redis")

def create_redis_health_check(redis_conn) -> bool:
    """
    Create a health check function for Redis.
    
    Args:
        redis_conn: Redis connection
        
    Returns:
        Boolean indicating if Redis is healthy
    """
    import redis
    
    try:
        # Check if connection is alive
        redis_conn.ping()
        
        # Try setting and getting a key
        test_key = "health_check_" + str(time.time())
        redis_conn.setex(test_key, 10, "OK")  # Key expires in 10 seconds
        value = redis_conn.get(test_key)
        
        return value == b"OK"
    except (redis.ConnectionError, redis.TimeoutError):
        return False
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        return False