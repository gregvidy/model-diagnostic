DECLARE @sampling_percentage FLOAT = {sampling_pct};

WITH tsf_credit AS (
    SELECT *
    FROM Transaction_Summary_Fraud_Hashed
    WHERE Channel = 'C03'
    AND Channel = 'C09'
)

, t_base AS (
    SELECT
		c03.PANNumber
        , c03.[Transaction Serial No]
		, c03.[Transaction Datetime]
		, c03.[Product Indicator]
		, c03.[Transaction Amount]
		, c03.[Card Billing Amount]
		, c03.MCC
		, c03.[Country Code]
		, c03.[Card Acceptor TerminalID] AS 'Card Acceptor Terminal ID'
		, c03.[Card Acceptor ID]
		, c03.[Terminal Owner] AS 'Card Acceptor Name'
		, c03.[Terminal CIty] AS 'Card Acceptor City'
		, c03.[Terminal State] AS 'Card Acceptor Region Code'
		, c03.[Terminal Country] AS 'Card Acceptor Country Code'
		, c03.[Currency Code]
        , tsf_credit.Confirmed
    FROM C03_Details_Hashed AS c03
    LEFT JOIN tsf_credit
        ON c03.[Transaction Serial No] = tsf_credit.[Transaction Serial No]

    UNION ALL

    SELECT
		c09.[Card No] AS PANNumber
        , c09.Transaction_Serial_No AS 'Transaction Serial No'
		, c09.[Transaction Datetime]
		, NULL AS 'Product Indicator'
		, c09.[Transaction Amount IDR] AS 'Transaction Amount'
		, NULL AS 'Card Billing Amount'
		, c09.MCC
		, c09.[Country Code]
		, c09.[Card Acceptor Terminal ID]
		, c09.[Card Acceptor Id Code] AS 'Card Acceptor ID'
		, c09.[Card Acceptor Name]
		, c09.[Card Acceptor City]
		, c09.[Card Acceptor Region]
		, c09.[Card Acceptor Country Code]
		, c09.[Transaction Currency Code] AS 'Currency Code'
        , tsf_credit.Confirmed
    FROM C09_Details_Hashed AS c09
    LEFT JOIN tsf_credit
        ON c09.[Transaction_Serial_No] = tsf_credit.[Transaction Serial No]
)

, t_clean AS (
	SELECT
		PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Product Indicator]
		, [Transaction Amount]
		, [Card Billing Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
		, [Card Acceptor ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
	FROM (
		SELECT *,
			ROW_NUMBER() OVER(
				PARTITION BY PANNumber
				ORDER BY CHECKSUM([Transaction Serial No])
			) AS rn,
			COUNT(*) OVER(
				PARTITION BY PANNumber
			) AS total
		FROM t_base
		WHERE Confirmed = 0 OR Confirmed IS NULL
	) AS sub
	WHERE rn <= total * (@sampling_percentage)
)

, t_fraud AS (
	SELECT
		PANNumber
        , [Transaction Serial No]
		, [Transaction Datetime]
		, [Product Indicator]
		, [Transaction Amount]
		, [Card Billing Amount]
		, MCC
		, [Country Code]
		, [Card Acceptor Terminal ID]
		, [Card Acceptor ID]
        , [Card Acceptor Name]
        , [Card Acceptor City]
        , [Card Acceptor Region]
        , [Card Acceptor Country Code]
		, [Currency Code]
		, Confirmed
	FROM t_base
	WHERE Confirmed = 1
)

, t_final AS (
	SELECT * FROM t_clean
	UNION ALL
	SELECT * FROM t_fraud
)

SELECT
    t_final.PANNumber
	, t_final.[Transaction Serial No]
    , t_final.[Transaction Datetime]
    , t_final.[Product Indicator]
    , t_final.[Transaction Amount]
    , t_final.[Card Billing Amount]
    , t_final.MCC
    , t_final.[Country Code]
    , t_final.[Card Acceptor Terminal ID]
    , t_final.[Card Acceptor ID]
    , t_final.[Card Acceptor Name]
    , t_final.[Card Acceptor City]
    , t_final.[Card Acceptor Region]
    , t_final.[Card Acceptor Country Code]
    , t_final.[Currency Code]
	, tscf.[Customer Age]
	, tscf.[Branch Name]
	, tscf.[Product Code]
	, tscf.[Customer Average Income]
	, tscf.[Card Status]
	, tscf.Balance
	, tscf.[Age of Open Account Transaction]
	, tscf.[Age of Active Card Transaction]
	, tscf.[Age of Open Account Active Card]
	, tscf.[Transfer To BDI Staff]
	, tscf.[BDI Staff]
	, tscf.[Account Open Year]
	, DATEDIFF(DAY, TRY_CAST(tscf.[Card Opening Date] AS datetime), TRY_CAST(tscf.[Transaction Datetime] AS datetime)) AS 'Age of Card Open to Transaction'
	, DATEDIFF(DAY, TRY_CAST(tscf.[Registration Date E-Channel?] AS datetime), TRY_CAST(tscf.[Transaction Datetime] AS datetime)) AS 'Age of EChannel Regist to Transaction'
	, tscf.[Is SDB Past Due]
	, tscf.[Flag Out Branch]
	, tscf.[Is TD Hold Amount]
	, tscf.[Is Program Hold Amount]
	, tscf.[Is Office Hour]
	, tscf.VALAS
	, tscf.[Transaction Load Time]
	, tscf.[Transaction Channel]
	, tscf.[Last Transaction Amount]
	, tscf.[Dr Last Transaction Amount]
	, tscf.[Cr Last Transaction Amount]
	, tscf.[Total Transaction Amount Last 1 Day]
	, tscf.[Dr Total Transaction Amount Last 1 Day]
	, tscf.[Cr Total Transaction Amount Last 1 Day]
	, tscf.[Total Transaction Amount Last 7 Days]
	, tscf.[Dr Total Transaction Amount Last 7 Days]
	, tscf.[Cr Total Transaction Amount Last 7 Days]
	, tscf.[Total Transaction Amount Last 1 Month?]
	, tscf.[Dr Total Transaction Amount Last 1 Month]
	, tscf.[Cr Total Transaction Amount Last 1 Month]
	, tscf.[Average Transaction Amount Last 1 Day]
	, tscf.[Average Transaction Amount Last 7 Days]
	, tscf.[Average Transaction Amount Last 1 Month?]
	, tscf.[Total Transaction Amount L15Mi]
	, tscf.[High Risk Customer]
	, tscf.[Product Name]
	, tscf.[Previous Account Status]
	, tscf.[Remark 4 (Tabungan)]
	, tscf.HoldAmountTB
	, tscf.[Remark 4 (Deposito)]
	, tscf.[Hold Amount DP]
	, tscf.[SDB Product Code]
	, tscf.[SDB Balance]
	, tscf.[CC Branch Code (Acc)]
	, tscf.[Branch Name (Acc)]
	, tscf.[CC Branch Code (Card)]
	, tscf.[Branch Name (Card)]
	, tscf.[CC Branch Code (Active Dormant)]
	, tscf.[Branch Name (Active Dormant)]
	, tscf.[Branch Name (Trx)]
	, tscf.[Is WhiteList?]
	, tscf.[Is Blacklist Account?]
	, tscf.[Is Blacklist Cust?]
	, tscf.[MNemonic Trx Desc]
	, tscf.[Product Code To]
	, tscf.[Access Code]
	, tscf.[SDB Status]
	, tscf.[SDB Last Payment Amount]
	, tscf.[Exclude Flag Txn Desc]
	, tscf.[Product Code (Deposito)]
	, tscf.[Is Merchant BDI?]
	, tscf.[Merchant Acc Name]
	, tscf.[Card Issuer Name]
	, tscf.[Total Trx Num L1D]
	, tscf.[Total Trx Num L7D]
	, tscf.[Is High Risk Country?]
	, tscf.[Is Blacklist Merchant?]
	, tscf.[Is WatchList Merchant?]
	, tscf.[Is High Risk MCC?]
	, tscf.[Customer Email]
	, tscf.[Credit Limit]
	, tscf.[Cash Credit Limit]
	, tscf.[Card Product]
	, tscf.[Customer No]
	, tscf.[Customer Sex]
	, tscf.[Is Blacklist Card No?]
	, tscf.[Is Valid Card?]
	, tscf.Currency
	, tscf.[POS Entry Mode]
	, tscf.[PIN Entry Capability]
	, tscf.[Is Card Expired?]
	, tscf.[Is 3DSecure?]
	, tscf.[Card Brand]
	, tscf.[Is WhiteList Merchant?]
	, tscf.[Is BDI Email?]
	, tscf.[Is WhiteList Card No?]
	, tscf.[Is WhiteList Account No?]
	, tscf.[Is Account BDI Employee?]
	, tscf.[Cod Access]
	, tscf.[Transaction Type]
	, tscf.[Total Transaction Amount L10Mi]
	, tscf.[Customer Type]
	, tscf.[Total Transaction Amount L5min]
	, tscf.TotalTrxAmountContactless
	, tscf.[Branch City]
	, tscf.[Branch Zip Code]
	, tscf.[Is Whitelist CardNo Open?]
	, tscf.DaysOfWhitelistCardNoOpen
	, tscf.[IsVAAccount?]
	, tscf.[Is Exclude Teminal Id?]
	, tscf.isBDICard
	, tscf.[Is Watchlist Card No?]
	, tscf.[Source Channel]
	, tscf.IsBlackListTID
	, tscf.IsFirstTrx3DS
	, tscf.[Card Data Input Capability]
	, tscf.CustomerNationality
	, tscf.[Total Transaction Amount Contactless per EOD]
	, tscf.IsTopUp
	, tscf.IsBillPayment
	, tscf.IsPaymentToVA
	, tscf.[Total Transaction Amount to Same EWallet L1D]
	, tscf.[Total Transaction Amount to Diff Wallet L1D]
	, tscf.[Total Transaction Amount to Same VA L1D]
	, tscf.[Total Transaction Amount to Diff VA L1D]
	, tscf.[Is High Risk Country Threshold Amount?]
	, tscf.[Is High Risk Currency?]
	, tscf.[Is High Risk Currency Threshold Amt?]
	, tscf.IsQRtrx
	, tscf.IsDCashTrx
	, tscf.IsFTTrx
	, tscf.IsDCashTrx1
	, tscf.IsFTTrx1
	, tscf.[CummTrxAmountQRIS 10Mins]
	, tscf.[CummTrxAmountDcash 15Mins]
	, tscf.[CummAmount Trf Last 10min]
	, tscf.[TrxQRIS Last 7D]
	, tscf.[TransactionTrf Last 7D]
	, tscf.[CummAmount Trf Last 20min]
	, tscf.[CountTrxQRIS2 10mins]
	, tscf.[CountTrxDcash 15Mins]
	, tscf.[CountTrxTrf2 10Mins]
	, tscf.[CountTrxVA Last 7D]
	, tscf.[CountTrxInqVA 15mins]
	, tscf.[Total Trx Num L15Mins]
	, tscf.[TOTAL AMOUNT LESS THAN 2 DAYS]
	, tscf.[Is Whitelist CardNo Previlage ?]
	, tscf.IsWhiteListTID
	, tscf.IsWatchListTID
	, tscf.[Time Difference between Current and Previous Transaction]
	, tscf.[Total Transaction Amount L30Mi]
	, tscf.IsBucket1WatchListCardNo
	, tscf.IsBucket1BlacklistCardNo
	, tscf.IsBucket1WhiteListCardNo
	, tscf.IsBucket2WhiteListMerchant
	, tscf.IsBucket2WatchListMerchant
	, tscf.IsBucket2BlacklistMerchant
	, tscf.CountTrx11PMto2AM
	, tscf.IsBucket3WhiteListMerchant
	, tscf.IsBucket3WatchListMerchant
	, tscf.IsBucket3WhiteListMID
	, tscf.IsBucket3WatchListMID
	, tscf.IsBucket3BlacklistMID
	, tscf.IsFacebookBlacklistMerchant
	, t_final.Confirmed
FROM t_final
LEFT JOIN Transaction_Summary_Calculations_Fraud_Hashed tscf
	ON t_final.[Transaction Serial No] = tscf.[Transaction Serial No]