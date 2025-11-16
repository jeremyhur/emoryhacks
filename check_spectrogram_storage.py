#!/usr/bin/env python3
"""
Script to verify that spectrogram URLs are being stored in Snowflake.
Run this after recording and analyzing audio to check if URLs are saved.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    logger.error("snowflake-connector-python not installed. Install it with: pip install snowflake-connector-python")
    sys.exit(1)

def check_spectrogram_storage():
    """Check if spectrogram URLs are being stored in Snowflake"""
    
    # Get Snowflake credentials from environment variables
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
    database = os.environ.get("SNOWFLAKE_DATABASE")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    table = os.environ.get("SNOWFLAKE_TABLE", "MONITORING_SESSIONS")
    
    if not all([account, user, password, warehouse, database]):
        logger.error("Missing Snowflake credentials. Please set the following environment variables:")
        logger.error("  - SNOWFLAKE_ACCOUNT")
        logger.error("  - SNOWFLAKE_USER")
        logger.error("  - SNOWFLAKE_PASSWORD")
        logger.error("  - SNOWFLAKE_WAREHOUSE")
        logger.error("  - SNOWFLAKE_DATABASE")
        return False
    
    try:
        logger.info("Connecting to Snowflake...")
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema,
            role='SYSADMIN'
        )
        
        cursor = conn.cursor()
        
        # Check if SPECTROGRAM_URL column exists
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Checking if SPECTROGRAM_URL column exists...")
        logger.info("="*60)
        
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table}'
              AND COLUMN_NAME = 'SPECTROGRAM_URL'
        """)
        
        column_info = cursor.fetchone()
        if column_info:
            logger.info(f"✓ SPECTROGRAM_URL column exists!")
            logger.info(f"  - Data Type: {column_info[1]}")
            logger.info(f"  - Max Length: {column_info[2]}")
        else:
            logger.warning("✗ SPECTROGRAM_URL column does NOT exist!")
            logger.warning("  Run this SQL to add it:")
            logger.warning(f"  ALTER TABLE {database}.{schema}.{table} ADD COLUMN IF NOT EXISTS SPECTROGRAM_URL VARCHAR(500);")
            cursor.close()
            conn.close()
            return False
        
        # Get statistics
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Getting storage statistics...")
        logger.info("="*60)
        
        cursor.execute(f"""
            SELECT 
                COUNT(*) as TOTAL_RECORDS,
                COUNT(SPECTROGRAM_URL) as RECORDS_WITH_SPECTROGRAM,
                COUNT(*) - COUNT(SPECTROGRAM_URL) as RECORDS_WITHOUT_SPECTROGRAM
            FROM {database}.{schema}.{table}
        """)
        
        stats = cursor.fetchone()
        total_records = stats[0]
        with_spectrogram = stats[1]
        without_spectrogram = stats[2]
        
        logger.info(f"Total Records: {total_records}")
        logger.info(f"Records with Spectrogram URL: {with_spectrogram}")
        logger.info(f"Records without Spectrogram URL: {without_spectrogram}")
        
        if total_records > 0:
            percentage = (with_spectrogram / total_records) * 100
            logger.info(f"Percentage with Spectrogram: {percentage:.2f}%")
        
        # Show recent records
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Recent records (last 5)...")
        logger.info("="*60)
        
        cursor.execute(f"""
            SELECT 
                SESSION_ID,
                TIMESTAMP,
                FATIGUE_STATUS,
                FILENAME,
                SPECTROGRAM_URL
            FROM {database}.{schema}.{table}
            ORDER BY TIMESTAMP DESC
            LIMIT 5
        """)
        
        records = cursor.fetchall()
        if records:
            for record in records:
                session_id, timestamp, fatigue_status, filename, spectrogram_url = record
                logger.info(f"\nSession ID: {session_id}")
                logger.info(f"  Timestamp: {timestamp}")
                logger.info(f"  Fatigue Status: {fatigue_status}")
                logger.info(f"  Audio URL: {filename[:80]}..." if filename and len(filename) > 80 else f"  Audio URL: {filename}")
                if spectrogram_url:
                    logger.info(f"  ✓ Spectrogram URL: {spectrogram_url[:80]}..." if len(spectrogram_url) > 80 else f"  ✓ Spectrogram URL: {spectrogram_url}")
                else:
                    logger.info(f"  ✗ Spectrogram URL: NULL (not uploaded)")
        else:
            logger.info("No records found in the table.")
        
        # Show sample spectrogram URLs
        if with_spectrogram > 0:
            logger.info("\n" + "="*60)
            logger.info("STEP 4: Sample Spectrogram URLs...")
            logger.info("="*60)
            
            cursor.execute(f"""
                SELECT 
                    SESSION_ID,
                    TIMESTAMP,
                    SPECTROGRAM_URL
                FROM {database}.{schema}.{table}
                WHERE SPECTROGRAM_URL IS NOT NULL
                ORDER BY TIMESTAMP DESC
                LIMIT 3
            """)
            
            urls = cursor.fetchall()
            for url_record in urls:
                session_id, timestamp, spectrogram_url = url_record
                logger.info(f"\nSession: {session_id}")
                logger.info(f"  Time: {timestamp}")
                logger.info(f"  URL: {spectrogram_url}")
        
        cursor.close()
        conn.close()
        
        logger.info("\n" + "="*60)
        logger.info("Verification complete!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking Snowflake: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = check_spectrogram_storage()
    sys.exit(0 if success else 1)

