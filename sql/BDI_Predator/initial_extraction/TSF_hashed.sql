/*
	2025-03-05 v0.1 Irfan
*/

DECLARE @ExtractDate DATETIME
SET @ExtractDate = GETDATE();

SELECT
	Transaction_Serial_No AS 'Transaction_Serial_No',
	Transaction_Datetime AS 'Transaction_Datetime',
	Channel AS 'Channel',
	HashBytes('SHA2_256', Group_By) AS 'Group_By',
	HashBytes('SHA2_256', Account_No) AS 'Account_No',
	HashBytes('SHA2_256', Customer_No) AS 'Customer_No',
	HashBytes('SHA2_256', Card_No) AS 'Card_No',
	Merchant_No AS 'Merchant_No',
	IP_Address AS 'IP_Address',
	Staff_Id AS 'Staff_Id',
	Transaction_Amount AS 'Transaction_Amount',
	Scorecard_Score AS 'Scorecard_Score',
	Grade AS 'Grade',
	Triggered_Rules AS 'Triggered_Rules',
	Assigned_User AS 'Assigned_User',
	Assigned_Team AS 'Assigned_Team',
	Assigned_Datetime AS 'Assigned_Datetime',
	Alert AS 'Alert',
	Risk_Level AS 'Risk_Level',
	Workflow_Status AS 'Workflow_Status',
	Action_Taken AS 'Action_Taken',
	Action_Datetime AS 'Action_Datetime',
	Reschedule_Datetime AS 'Reschedule_Datetime',
	Confirmed AS 'Confirmed',
	Case_No AS 'Case_No',
	Merchant_Category_Code AS 'Merchant_Category_Code',
	Country_Code AS 'Country_Code',
	Currency_Code AS 'Currency_Code',
	Rule_Decision AS 'Rule_Decision'
INTO [DataExtractionDB].[dbo].[Transaction_Summary_Fraud_Hashed]
FROM [BDI-PRD-PDTORDB].[Predator].[dbo].[Transaction_Summary_Fraud] WITH (NOLOCK) WHERE Transaction_Datetime <= @ExtractDate