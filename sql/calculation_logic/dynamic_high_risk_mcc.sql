WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
    FROM C03_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(PANNumber, '-999') AS PANNumber,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime,
    FROM C09_Channel
),

cte_joined AS (
    SELECT
        cte_base.Transaction_Serial_No,
        cte_base.Transaction_Datetime,
        cte_base.MCC,
        tsf.Confirmed
    FROM cte_base
    LEFT JOIN Transaction_Summary_Fraud AS tsf
        ON cte_base.Transaction_Serial_No = tsf.Transaction_Serial_No
),

dynamic_high_risk_mcc_l30d AS (
    SELECT TOP 10
        MCC,
        COUNT(*) AS Fraud_Occurence_L30D
    FROM cte_joined
    WHERE
        Confirmed = 1 AND
        Transaction_Datetime >= DATEADD(DAY, -30, GETDATE())
    GROUP BY MCC
)

SELECT
    cte_base.Transaction_Serial_No,
    cte_base.PANNumber,
    CASE WHEN dynamic_mcc.MCC IS NULL THEN 0
        ELSE 1
        END AS IsTop10HighRiskMCCLast30D
FROM cte_base
LEFT JOIN dynamic_high_risk_mcc_l30d AS dynamic_mcc
    ON dynamic_mcc.MCC = cte_base.MCC