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

SELECT
    cte_base.Transaction_Serial_No,
    cte_base.Debit_No,
    COALESCE(tscf.HighRiskCustomer, "__missing__") AS HighRiskCustomer,
    COALESCE(SAFE_CAST(CAST(tscf.POSMode AS INT) AS STRING), "__missing__") AS POSMode,
    COALESCE(tscf.CustomerAge, -999) AS CustomerAge,
    COALESCE(tscf.CountTrxTrf, 0) AS CountTrxTrf,
FROM cte_base
LEFT JOIN Transaction_Summary_Calculation_Fraud AS tscf
    ON cte_base.Transaction_Serial_No = tscf.Transaction_Serial_No