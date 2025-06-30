WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        PANNumber,
        MCC,
        Transaction_Datetime,
    FROM C03_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        PANNumber,
        MCC,
        Transaction_Datetime,
    FROM C09_Channel
),

t_min_date AS (
    SELECT
        PANNumber,
        MCC,
        MIN(Transaction_Datetime) AS first_txn_date
    FROM cte_base
    GROUP BY PANNumber, MCC
),

SELECT
    Transaction_Serial_No,
    PANNumber,
    MCC,
    Transaction_Datetime,
    DATE_DIFF(Transaction_Datetime, first_txn_date, DAY) AS TimeFirstTxnToCurrentMCC 
FROM cte_base
LEFT JOIN t_min_date
    ON cte_base.PANNumber = t_min_date.PANNumber
    AND cte_base.MCC = t_min_date.MCC