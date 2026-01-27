DECLARE @sampling_percentage FLOAT = {sampling_pct};

WITH tsf_debit AS (
    SELECT
    	tsf.Transaction_Serial_No
    	, CASE WHEN CAST(cust_dispute.[Transaction Serial No] AS bigint) = tsf.Transaction_Serial_No THEN 1
    			ELSE tsf.Confirmed
    			END AS Confirmed
    FROM Transaction_Summary_Fraud_Hashed AS tsf
    LEFT JOIN tempDBpredator.dbo.C06C10 AS cust_dispute
    	ON CAST(cust_dispute.[Transaction Serial No] AS bigint) = tsf.Transaction_Serial_No
    WHERE tsf.Channel = 'C06'
    OR tsf.Channel = 'C10'
)

, t_base AS (
    SELECT
		c06.[Debit Account No] AS Debit_No
        , c06.[Transaction_Serial_No] AS 'Transaction Serial No'
		, c06.[Transaction Datetime]
		, c06.[TransactionAmount] AS 'Transaction Amount'
		, c06.MCC
		, c06.CardAcceptorCountryCode AS 'Country Code'
		, c06.CardAcceptorTerminalID AS 'Card Acceptor Terminal ID'
		, c06.CardAcceptorCity AS 'Card Acceptor City' 
		, c06.CardAcceptorName AS 'Card Acceptor Name' 
		, c06.CardAcceptorRegionCode AS 'Card Acceptor Region' 
		, c06.CardAcceptorCountryCode AS 'Card Acceptor Country Code' 
		, c06.TransactionCurrencyCode AS 'Currency Code'
        , tsf_debit.Confirmed
    FROM C06_Details_Hashed AS c06
    LEFT JOIN tsf_debit
        ON c06.[Transaction_Serial_No] = tsf_debit.Transaction_Serial_No

    UNION ALL

    SELECT
		c10.[Debit Account No] AS Debit_No
        , c10.[Transaction_Serial_No] AS 'Transaction Serial No'
		, c10.[Transaction Datetime]
		, c10.[Transaction Amount]
		, c10.MCC
		, c10.CardAcceptorCountryCode AS 'Country Code'
		, c10.CardAcceptorTerminalID AS 'Card Acceptor Terminal ID'
		, c10.CardAcceptorCity AS 'Card Acceptor City' 
		, c10.CardAcceptorName AS 'Card Acceptor Name' 
		, c10.CardAcceptorRegionCode AS 'Card Acceptor Region' 
		, c10.CardAcceptorCountryCode AS 'Card Acceptor Country Code' 
		, '360' AS 'Currency Code'
        , tsf_debit.Confirmed
    FROM C10_Details_Hashed AS c10
    LEFT JOIN tsf_debit
        ON c10.[Transaction_Serial_No] = tsf_debit.Transaction_Serial_No
)

, t_list_card_ever_alert AS (
    SELECT DISTINCT Debit_No
    FROM t_base
    WHERE Confirmed = 1 OR Confirmed = 0
)

, t_list_card_never_alert AS (
    SELECT DISTINCT t_base.Debit_No
    FROM t_base
    LEFT JOIN t_list_card_ever_alert
        ON t_base.Debit_No = t_list_card_ever_alert.Debit_No
    WHERE t_list_card_ever_alert.Debit_No IS NULL
)

, t_sample AS (
    SELECT Debit_No
    FROM (
        SELECT
            Debit_No
            , ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn
            , COUNT(*) OVER () AS total
        FROM t_list_card_never_alert
    ) AS sub
    WHERE rn <= total * @sampling_percentage
)

, t_all_clean_population AS (
    SELECT
		t_base.Debit_No
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Transaction Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
    FROM t_base
    INNER JOIN t_sample
        ON t_base.Debit_No = t_sample.Debit_No
)

SELECT
    t_final.Debit_No
    , t_final.[Transaction Serial No]
    , t_final.[Transaction Datetime]
    , t_final.[Transaction Amount]
    , t_final.MCC
    , t_final.[Country Code]
    , t_final.[Card Acceptor Terminal ID]
    , t_final.[Card Acceptor Name]
    , t_final.[Card Acceptor City]
    , t_final.[Card Acceptor Region]
    , t_final.[Card Acceptor Country Code]
    , t_final.[Currency Code]
	, tscf.CustomerAge
	, tscf.AccountStatus
	, tscf.CustomerAvgIncome
	, tscf.CardStatus
	, tscf.Balance
	, tscf.AgeOfOpenAcctTxn
	, tscf.AgeOfActiveCardTxn
	, tscf.AgeOfOpenAcctActiveCard
	, tscf.TrfToBDIStaff
	, tscf.BDIStaff
	, tscf.AgeOfRegDateTxn
	, tscf.IsSDBPastDue
	, tscf.FlagOutBranch
	, tscf.isTDHoldAmount
	, tscf.isProgramHoldAmount
	, tscf.IsOfficeHour
	, tscf.VALAS
	, tscf.ChannelTrx
	, tscf.LastTrxAmount
	, tscf.DrLastTrxAmount
	, tscf.CrLastTrxAmount
	, tscf.TotalTrxAmountL1D
	, tscf.DrTotalTrxAmountL1D
	, tscf.CrTotalTrxAmountL1D
	, tscf.TotalTrxAmountL7D
	, tscf.DrTotalTrxAmountL7D
	, tscf.CrTotalTrxAmountL7D
	, tscf.TotalTrxAmountL1M
	, tscf.DrTotalTrxAmountL1M
	, tscf.CrTotalTrxAmountL1M
	, tscf.AvgTrxAmountL1D
	, tscf.AvgTrxAmountL7D
	, tscf.AvgTrxAmountL1M
	, tscf.TotalTrxAmount15Mi
	, tscf.HIghRiskCustomer
	, tscf.ProductName
	, tscf.PrevAccStatus
	, tscf.HoldAmountTB
	, tscf.HoldAmountDP
	, tscf.SDBBalance
	, tscf.CCBrnAccount
	, tscf.CCBrnReactivedAccDormant
	, tscf.BrnNameReactivedAccDormant
	, tscf.TrxBranchName
	, tscf.IsWhiteList
	, tscf.IsBlacklistAccount
	, tscf.IsBlacklistCust
	, tscf.ProductCodeTo
	, tscf.AccessCode
	, tscf.SDBBoxNo
	, tscf.SDBStatus
	, tscf.SDBLastPaymentAmount
	, tscf.FlagTxnDesc
	, tscf.ProductCodeDep
	, tscf.IsMerchantBDI
	, tscf.NumOfTrxL1D
	, tscf.NumOfTrxL7D
	, tscf.IsHighRiskCountry
	, tscf.IsBlacklistMerchant
	, tscf.IsWatchListMerchant
	, tscf.IsHighRiskMCC
	, tscf.CreditLimit
	, tscf.CashCreditLimit
	, tscf.CardProduct
	, tscf.CustomerSex
	, tscf.IsBlacklistCardNo
	, tscf.IsValidCard
	, tscf.Currency
	, tscf.POSMode
	, tscf.PINEntryCapability
	, tscf.IsCardExpired
	, tscf.Is3DSecure
	, tscf.IsWhiteListMerchant
	, tscf.IsBDIEmail
	, tscf.IsWhiteListCardNo
	, tscf.IsWhiteListAccountNo
	, tscf.IsWatchListAccountNo
	, tscf.IsAccountBDIEmployee
	, tscf.CodAccess
	, tscf.TransactionType
	, tscf.TotalTrxAmount10Mi
	, tscf.CustomerType
	, tscf.TotalTrxAmountL5min
	, tscf.TotalTrxAmountContactless
	, tscf.IsWhitelistCardNoOpen
	, tscf.DaysOfWhitelistCardNoOpen
	, tscf.IsVAAccount
	, tscf.IsExcludeTeminalId
	, tscf.isBDICard
	, tscf.IsWatchlistCardNo
	, tscf.SourceChannel
	, tscf.IsBlackListTID
	, tscf.IsFirstTrx3DS
	, tscf.VCardDataInputCapability
	, tscf.CustomerNationality
	, tscf.TotalTrxAmountContactlessPerDay
	, tscf.IsTopUp
	, tscf.IsBillPayment
	, tscf.IsPaymentToVA
	, tscf.TotalTrxAmountTopUp1
	, tscf.TotalTrxAmountTopUp2
	, tscf.TotalTrxAmountVA1
	, tscf.TotalTrxAmountVA2
	, tscf.CustomerEwaletNumber
	, tscf.IsHighRiskCountryThresholdAmt
	, tscf.IsHighRiskCurrency
	, tscf.IsHighRiskCurrencyThresholdAmt
	, tscf.IsQRtrx
	, tscf.IsDCashTrx
	, tscf.IsFTTrx
	, tscf.TotalTrxAmountQRIS
	, tscf.TotalTrxAmountDcash
	, tscf.TotalTrxAmountTrf
	, tscf.CountTrxQRIS
	, tscf.CountTrxTrf
	, tscf.TotalTrxAmountTrf2
	, tscf.CountTrxQRIS2
	, tscf.CountTrxDcash
	, tscf.CountTrxTrf2
	, tscf.NumOfTrxL15Min
	, tscf.IsWhitelistCardNoPrevilage
	, tscf.IsWhiteListTID
	, tscf.IsWatchListTID
	, tscf.TimeDiffCurrPrev
	, tscf.TotalTrxAmount30Mi
	, tscf.IsBucket1WatchListCardNo
	, tscf.IsBucket1BlacklistCardNo
	, tscf.IsBucket1WhiteListCardNo
	, tscf.IsBucket2WhiteListMerchant
	, tscf.IsBucket2WatchListMerchant
	, tscf.IsBucket2BlacklistMerchant
	, tscf.CountTrxEOD
	, t_final.Confirmed
FROM t_all_clean_population AS t_final
LEFT JOIN Transaction_Summary_Calculations_Fraud_Hashed tscf
	ON t_final.[Transaction Serial No] = tscf.Transaction_Serial_No