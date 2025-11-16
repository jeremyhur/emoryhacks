-- ============================================
-- VERIFY AND FIX SNOWFLAKE TABLE
-- Run these commands to check and fix your table
-- ============================================

-- STEP 1: Check your current context
SELECT CURRENT_DATABASE(), CURRENT_SCHEMA();

-- STEP 2: Use your database and schema (replace with your values)
-- USE DATABASE YOUR_DATABASE;
-- USE SCHEMA YOUR_SCHEMA;

-- STEP 3: Check if the table exists
SHOW TABLES LIKE 'MONITORING_SESSIONS';

-- STEP 4: If the table exists, check its structure
DESCRIBE TABLE MONITORING_SESSIONS;

-- STEP 5: If the table doesn't exist OR has wrong columns, drop and recreate it
-- WARNING: This will delete all existing data!
-- DROP TABLE IF EXISTS MONITORING_SESSIONS;

-- STEP 6: Create the table with the correct structure
CREATE TABLE IF NOT EXISTS MONITORING_SESSIONS (
    SESSION_ID VARCHAR(36) PRIMARY KEY,
    TIMESTAMP TIMESTAMP_NTZ NOT NULL,
    FATIGUE_STATUS VARCHAR(20) NOT NULL,
    TRANSCRIPTION TEXT,
    DOMINANT_EMOTION VARCHAR(20),
    NEUTRAL_PROBABILITY FLOAT,
    HAPPY_PROBABILITY FLOAT,
    ANGRY_PROBABILITY FLOAT,
    SAD_PROBABILITY FLOAT,
    ACTIVE_PERCENTAGE FLOAT,
    FATIGUED_PERCENTAGE FLOAT,
    AI_SUPERVISOR_REPORT TEXT,
    FILENAME VARCHAR(255)
);

-- STEP 7: Verify the table structure again
DESCRIBE TABLE MONITORING_SESSIONS;

-- You should see all 13 columns listed above
-- If you see them, the table is ready!

