#!/usr/bin/env python3
"""
Simple Redis connection tester to diagnose issues.
Run this script to test connectivity to Redis.
"""

import sys
import socket
import time
import os

def test_redis_connection(host, port, retries=3):
    """Test direct TCP connection to Redis."""
    print(f"Testing TCP connection to Redis at {host}:{port}")
    
    for attempt in range(retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✅ Successfully connected to Redis at {host}:{port}")
                return True
            else:
                print(f"❌ TCP connection failed to {host}:{port} - attempt {attempt+1}/{retries}")
        except Exception as e:
            print(f"❌ Error connecting to {host}:{port}: {e}")
        
        if attempt < retries - 1:
            print(f"Waiting 1 second before retry...")
            time.sleep(1)
    
    print(f"❌ Failed to establish TCP connection after {retries} attempts")
    return False

def test_redis_ping(host, port, retries=3):
    """Test Redis PING command."""
    try:
        import redis
    except ImportError:
        print("❌ Redis package not installed. Run: pip install redis")
        return False
    
    print(f"Testing Redis PING command at {host}:{port}")
    
    for attempt in range(retries):
        try:
            client = redis.Redis(
                host=host, 
                port=port, 
                socket_timeout=5,
                socket_connect_timeout=5
            )
            response = client.ping()
            
            if response:
                print(f"✅ Redis responded to PING at {host}:{port}")
                return True
            else:
                print(f"❌ Redis PING failed at {host}:{port} - attempt {attempt+1}/{retries}")
        except Exception as e:
            print(f"❌ Redis PING error at {host}:{port}: {e}")
        
        if attempt < retries - 1:
            print(f"Waiting 1 second before retry...")
            time.sleep(1)
    
    print(f"❌ Failed Redis PING after {retries} attempts")
    return False

def test_redis_set_get(host, port, retries=3):
    """Test Redis SET/GET commands."""
    try:
        import redis
    except ImportError:
        print("❌ Redis package not installed. Run: pip install redis")
        return False
    
    print(f"Testing Redis SET/GET at {host}:{port}")
    
    for attempt in range(retries):
        try:
            client = redis.Redis(
                host=host, 
                port=port, 
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            test_key = f"test_key_{time.time()}"
            test_value = "test_value"
            
            # Try SET
            client.set(test_key, test_value)
            print(f"✅ Redis SET worked at {host}:{port}")
            
            # Try GET
            value = client.get(test_key)
            
            if value.decode() == test_value:
                print(f"✅ Redis GET worked at {host}:{port}")
                # Cleanup
                client.delete(test_key)
                return True
            else:
                print(f"❌ Redis GET failed at {host}:{port} - attempt {attempt+1}/{retries}")
        except Exception as e:
            print(f"❌ Redis SET/GET error at {host}:{port}: {e}")
        
        if attempt < retries - 1:
            print(f"Waiting 1 second before retry...")
            time.sleep(1)
    
    print(f"❌ Failed Redis SET/GET after {retries} attempts")
    return False

def print_network_info():
    """Print basic network information."""
    print("\n--- Network Information ---")
    
    # Current hostname
    try:
        hostname = socket.gethostname()
        print(f"Hostname: {hostname}")
    except Exception as e:
        print(f"Error getting hostname: {e}")
    
    # DNS resolution test for Redis host
    try:
        redis_host = "multiagent-redis"
        ip = socket.gethostbyname(redis_host)
        print(f"DNS resolution: {redis_host} resolves to {ip}")
    except socket.gaierror:
        print(f"❌ DNS resolution failed for {redis_host}")
    
    # Also try with just 'redis'
    try:
        redis_host = "redis"
        ip = socket.gethostbyname(redis_host)
        print(f"DNS resolution: {redis_host} resolves to {ip}")
    except socket.gaierror:
        print(f"❌ DNS resolution failed for {redis_host}")

def print_config_info():
    """Print configuration information."""
    print("\n--- Configuration Information ---")
    
    # Check environment variables
    redis_uri = os.environ.get("REDIS_URI", "Not set")
    print(f"REDIS_URI environment variable: {redis_uri}")
    
    # Try to parse Redis URI
    if redis_uri != "Not set":
        try:
            if "://" in redis_uri:
                parts = redis_uri.split("://")[1].split(":")
                host = parts[0]
                port_db = parts[1].split("/")
                port = port_db[0]
                db = port_db[1] if len(port_db) > 1 else "0"
                print(f"Parsed from REDIS_URI: host={host}, port={port}, db={db}")
        except Exception as e:
            print(f"Error parsing REDIS_URI: {e}")

def main():
    """Main function to run all tests."""
    print("Redis Connection Test Script")
    print("===========================")
    
    print_network_info()
    print_config_info()
    
    # Default Redis host and port
    default_host = "multiagent-redis"
    default_port = 6379
    
    # Get host and port from arguments or environment
    redis_uri = os.environ.get("REDIS_URI", "")
    if redis_uri and "://" in redis_uri:
        try:
            parts = redis_uri.split("://")[1].split(":")
            host = parts[0]
            port = int(parts[1].split("/")[0])
        except Exception:
            host = default_host
            port = default_port
    else:
        host = default_host
        port = default_port
    
    print(f"\nTesting Redis connection to {host}:{port}")
    
    # Run tests
    tcp_ok = test_redis_connection(host, port)
    
    if tcp_ok:
        ping_ok = test_redis_ping(host, port)
        if ping_ok:
            set_get_ok = test_redis_set_get(host, port)
    
    # Try alternate hostnames if primary tests fail
    alternate_hosts = ["redis", "localhost", "127.0.0.1"]
    if host not in alternate_hosts and not tcp_ok:
        print("\nTrying alternate hostnames...")
        
        for alt_host in alternate_hosts:
            print(f"\nTesting alternate host: {alt_host}")
            if test_redis_connection(alt_host, port):
                test_redis_ping(alt_host, port)
                test_redis_set_get(alt_host, port)
                
                # Suggest configuration change
                print(f"\n💡 Consider changing REDIS_URI to redis://{alt_host}:{port}/0")
                break

if __name__ == "__main__":
    main()