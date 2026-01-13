WITH cte_base AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
    FROM C06_Channel
    UNION ALL
    SELECT
        Transaction_Serial_No,
        COALESCE(Debit_No, '-999') AS Debit_No,
        Transaction_Datetime,
    FROM C10_Channel
),

, cte_tscf AS (
    SELECT
        Transaction_Serial_No,
        COALESCE(HighRiskCustomer, "__missing__") AS HighRiskCustomer,
        COALESCE(CustomerAge, -999) AS CustomerAge,
        CASE WHEN SAFE_CAST(CAST(tscf.POSMode AS INT) AS STRING) NOT IN ('0', '1', '2', '5', '7', '9') THEN "__missing__"
             WHEN SAFE_CAST(CAST(tscf.POSMode AS INT) AS STRING) IS NULL THEN "__missing__"
        ELSE SAFE_CAST(CAST(tscf.POSMode AS INT) AS STRING)
        END AS POSMode,
        COALESCE(CountTrxTrf, 0) AS CountTrxTrf
    FROM Transaction_Summary_Calculation_Fraud
)

SELECT
    cte_base.Transaction_Serial_No,
    cte_base.Debit_No,
    cte_tscf.HighRiskCustomer,
    cte_tscf.POSMode,
    cte_tscf.CustomerAge,
    cte_tscf.CountTrxTrf,
FROM cte_base
LEFT JOIN cte_tscf
    ON cte_base.Transaction_Serial_No = cte_tscf.Transaction_Serial_No