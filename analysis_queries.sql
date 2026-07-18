-- =====================================================================
-- Software Release Risk — SQL Analysis Queries
-- Table: releases (loaded from software_release_risk_dataset.csv)
-- Works in SQLite out of the box. For MySQL/Postgres/Power BI, adjust
-- ROUND()/CAST() syntax only if your engine complains — logic is the same.
-- =====================================================================

-- 1. Overall risk level distribution
SELECT
    Risk_Level,
    COUNT(*) AS Release_Count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM releases), 2) AS Percentage
FROM releases
GROUP BY Risk_Level
ORDER BY Release_Count DESC;

-- 2. Average metrics by risk level (what actually drives risk)
SELECT
    Risk_Level,
    ROUND(AVG(Commits), 1)                  AS Avg_Commits,
    ROUND(AVG(Bugs_Reported), 1)            AS Avg_Bugs,
    ROUND(AVG(Critical_Bugs), 1)            AS Avg_Critical_Bugs,
    ROUND(AVG(Test_Coverage), 1)            AS Avg_Test_Coverage,
    ROUND(AVG(Failed_Test_Cases), 1)        AS Avg_Failed_Tests,
    ROUND(AVG(Code_Churn), 0)               AS Avg_Code_Churn,
    ROUND(AVG(Cyclomatic_Complexity), 1)    AS Avg_Complexity,
    ROUND(AVG(Security_Vulnerabilities), 2) AS Avg_Security_Vulns,
    ROUND(AVG(Developer_Experience), 1)     AS Avg_Dev_Experience
FROM releases
GROUP BY Risk_Level
ORDER BY Avg_Critical_Bugs DESC;

-- 3. High-risk releases with weakest test coverage (candidates to flag)
SELECT
    Release_ID, Test_Coverage, Failed_Test_Cases, Critical_Bugs, Security_Vulnerabilities
FROM releases
WHERE Risk_Level = 'High'
ORDER BY Test_Coverage ASC
LIMIT 20;

-- 4. Test coverage buckets vs risk level (does coverage really matter?)
SELECT
    CASE
        WHEN Test_Coverage < 50 THEN '0-50%'
        WHEN Test_Coverage < 70 THEN '50-70%'
        WHEN Test_Coverage < 85 THEN '70-85%'
        ELSE '85-100%'
    END AS Coverage_Bucket,
    Risk_Level,
    COUNT(*) AS Release_Count
FROM releases
GROUP BY Coverage_Bucket, Risk_Level
ORDER BY Coverage_Bucket, Risk_Level;

-- 5. Developer experience vs critical bugs (junior team risk check)
SELECT
    CASE
        WHEN Developer_Experience < 2 THEN '0-2 yrs'
        WHEN Developer_Experience < 5 THEN '2-5 yrs'
        WHEN Developer_Experience < 10 THEN '5-10 yrs'
        ELSE '10+ yrs'
    END AS Experience_Bucket,
    ROUND(AVG(Critical_Bugs), 2) AS Avg_Critical_Bugs,
    ROUND(AVG(Security_Vulnerabilities), 2) AS Avg_Security_Vulns,
    COUNT(*) AS Release_Count
FROM releases
GROUP BY Experience_Bucket
ORDER BY Experience_Bucket;

-- 6. Code churn vs deployment frequency, correlated with risk
SELECT
    Risk_Level,
    ROUND(AVG(Code_Churn), 0) AS Avg_Code_Churn,
    ROUND(AVG(Files_Changed), 1) AS Avg_Files_Changed,
    ROUND(AVG(Deployment_Frequency), 1) AS Avg_Deploy_Frequency,
    ROUND(AVG(Build_Time_Minutes), 1) AS Avg_Build_Time
FROM releases
GROUP BY Risk_Level;

-- 7. Top 10 riskiest-looking releases overall (multi-factor sort)
SELECT
    Release_ID, Risk_Level, Critical_Bugs, Security_Vulnerabilities,
    Failed_Test_Cases, Test_Coverage
FROM releases
ORDER BY Critical_Bugs DESC, Security_Vulnerabilities DESC, Failed_Test_Cases DESC
LIMIT 10;

-- 8. Security vulnerability exposure by risk level
SELECT
    Risk_Level,
    SUM(Security_Vulnerabilities) AS Total_Security_Vulns,
    ROUND(AVG(Security_Vulnerabilities), 2) AS Avg_Security_Vulns,
    MAX(Security_Vulnerabilities) AS Max_Security_Vulns
FROM releases
GROUP BY Risk_Level
ORDER BY Total_Security_Vulns DESC;
