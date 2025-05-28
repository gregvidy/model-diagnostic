/*
	2025-05-16 v0.2 Adam
*/

DECLARE @ExtractDate DATETIME
SET @ExtractDate = GETDATE();

SELECT
	Transaction_Serial_No AS 'Transaction_Serial_No',
	MTI AS 'MTI',
	PrimaryBitMap AS 'Primary Bit Map',
	SecondaryBitMap AS 'Secondary Bit Map',
	HashBytes('SHA2_256', PANNumber) AS 'PANNumber',
	ProcessingCode AS 'Processing Code',
	TransactionAmount AS 'Transaction Amount',
	TransmissionDateTime AS 'Transmission Date Time',
	STAN AS 'STAN',
	Transaction_Datetime AS 'Transaction Datetime',
	SettlementDate AS 'Settlement Date',
	CaptureDate AS 'Capture Date',
	MCC AS 'MCC',
	CardInputMode AS 'CardInputMode',
	PINCaptureCapability AS 'PIN Capture Capability',
	AcquiringID AS 'Acquiring ID',
	RRN AS 'Retrival Reference Number',
	CardAcceptorTerminalID AS 'CardAcceptorTerminalID',
	HashBytes('SHA2_256', CardAcceptorName) AS 'CardAcceptorName',
	HashBytes('SHA2_256', CardAcceptorStreet) AS 'CardAcceptorStreet',
	CardAcceptorCity AS 'CardAcceptorCity',
	CardAcceptorPostalCode AS 'CardAcceptorPostalCode',
	CardAcceptorRegionCode AS 'CardAcceptorRegionCode',
	CardAcceptorCountryCode AS 'CardAcceptorCountryCode',
	AdditionalData AS 'Additional Data',
	OriginalSTAN AS 'OriginalSTAN',
	OriginalTransmissionDatetime AS 'OriginalTransmissionDatetime',
	BillCompanyID AS 'BillCompanyID',
	BIllNumber AS 'BIllNumber',
	HashBytes('SHA2_256', BillConsumerNumber) AS 'BillConsumerNumber',
	BillRefNo1 AS 'BillRefNo1',
	BillRefNo2 AS 'BillRefNo2',
	HashBytes('SHA2_256', AccountIdentification1) AS 'Debit Account No',
	HashBytes('SHA2_256', AccountIdentification2) AS 'Credit Account No',
	ORFTContraBankID AS 'ORFTContraBankID',
	ORFTContraBankAccNo AS 'ORFTContraBankAccNo',
	HashBytes('SHA2_256', ReferalPhoneNumber) AS 'ReferalPhoneNumber',
	HashBytes('SHA2_256', MemberNumber) AS 'MemberNumber',
	POSTerminalID AS 'POSTerminalID',
	SICCode AS 'SICCode'
INTO [DataExtractionDB].[dbo].[C10_Details_Hashed]
FROM [BDI-PRD-PDTORDB].[Predator].[dbo].[C10_Details] WITH (NOLOCK) WHERE Transaction_Datetime <= @ExtractDate