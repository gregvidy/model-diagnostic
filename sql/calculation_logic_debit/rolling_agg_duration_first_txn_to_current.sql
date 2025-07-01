WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime
    FROM C06_Channel

    UNION ALL

    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        COALESCE(MCC, '-999') AS MCC,
        Transaction_Datetime
    FROM C10_Channel
),

t_min_date AS (
    SELECT
        Debit_No,
        MCC,
        MIN(Transaction_Datetime) AS first_txn_date
    FROM cte_base
    GROUP BY Debit_No, MCC
),

cte_joined AS (
    SELECT
        A.Transaction_Serial_No,
        A.Debit_No,
        A.MCC,
        A.Transaction_Datetime,
        DATEDIFF(MINUTE, B.first_txn_date, A.Transaction_Datetime) AS TimeFirstTxnToCurrentMCC
    FROM cte_base AS A
    LEFT JOIN t_min_date AS B
        ON A.Debit_No = B.Debit_No
        AND A.MCC = B.MCC
)

SELECT 
    A.Transaction_Serial_No,
    A.Debit_No,
    A.MCC,
    A.Transaction_Datetime,
    A.TimeFirstTxnToCurrentMCC,

    -- Rolling avg over last 30 days, excluding current row
    (
        SELECT AVG(B.TimeFirstTxnToCurrentMCC * 1.0)
        FROM cte_joined AS B
        WHERE 
            B.Debit_No = A.Debit_No AND
            B.MCC = A.MCC AND
            B.Transaction_Datetime < A.Transaction_Datetime AND
            B.Transaction_Datetime >= DATEADD(DAY, -30, A.Transaction_Datetime)
    ) AS AvgTimeFirstTxnToCurrentMCC_Last30D

FROM cte_joined AS A;
