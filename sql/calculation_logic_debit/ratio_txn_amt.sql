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

cte_joined AS (
    SELECT
        cte_base.Transaction_Serial_No,
        cte_base.Debit_No,
        tscf.TotalTrxAmount10Mi,
        tscf.TotalTrxAmountL5min,
        tscf.TotalTrxAmount15Mi,
        tscf.TotalTrxAmount1LD
    FROM cte_base
    LEFT JOIN Transaction_Summary_Calculation_Fraud AS tscf
        ON tscf.Transaction_Serial_No = cte_base.Transaction_Serial_No
)

SELECT
    Transaction_Serial_No,
    Debit_No,
    TotalTrxAmount15Mi,
    TotalTrxAmount10Mi,
    TotalTrxAmountL5min,
    TotalTrxAmountL1D,

    -- Safe division: if denominator is 0 or NULL, impute ratio as 0
    ISNULL(
        TotalTrxAmount1LD / 
        NULLIF(TotalTrxAmount15Mi, 0),
        0        
    ) AS RatioTrxAmount_L1DL15min,
    ISNULL(
        TotalTrxAmount1LD / 
        NULLIF(TotalTrxAmount10Mi, 0),
        0        
    ) AS RatioTrxAmount_L1DL10min,
    ISNULL(
        TotalTrxAmount1LD / 
        NULLIF(TotalTrxAmountL5min, 0),
        0        
    ) AS RatioTrxAmount_L1DL5min,
FROM cte_joined