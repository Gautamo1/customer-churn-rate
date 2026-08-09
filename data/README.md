# Dataset Documentation

## Source

The dataset used in this project is:

**Predictive Analytics for Customer Churn Dataset**

Source: Kaggle — Safrin03

[Kaggle Dataset](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset?utm_source=chatgpt.com)

The dataset contains customer-level information from a subscription-based content streaming service. It includes account information, billing details, subscription preferences, viewing behavior, customer preferences, and support activity.

## Dataset Purpose

The dataset is used to build a machine learning model that predicts whether a customer is likely to churn.

The target variable is:

* `Churn = 1` → Customer has churned
* `Churn = 0` → Customer has not churned

## Features

| Column                     | Type       | Data Type | Description                                                                   |
| -------------------------- | ---------- | --------- | ----------------------------------------------------------------------------- |
| `AccountAge`               | Feature    | Integer   | Age of the user's account in months.                                          |
| `MonthlyCharges`           | Feature    | Float     | Amount charged to the user on a monthly basis.                                |
| `TotalCharges`             | Feature    | Float     | Total charges incurred by the user over the account's lifetime.               |
| `SubscriptionType`         | Feature    | Object    | Type of subscription: Basic, Standard, or Premium.                            |
| `PaymentMethod`            | Feature    | String    | Payment method used by the user.                                              |
| `PaperlessBilling`         | Feature    | String    | Whether the user has opted for paperless billing: Yes or No.                  |
| `ContentType`              | Feature    | String    | Type of content preferred: Movies, TV Shows, or Both.                         |
| `MultiDeviceAccess`        | Feature    | String    | Whether the user has access on multiple devices: Yes or No.                   |
| `DeviceRegistered`         | Feature    | String    | Type of registered device: TV, Mobile, Tablet, or Computer.                   |
| `ViewingHoursPerWeek`      | Feature    | Float     | Number of hours the user watches content per week.                            |
| `AverageViewingDuration`   | Feature    | Float     | Average duration of each viewing session in minutes.                          |
| `ContentDownloadsPerMonth` | Feature    | Integer   | Number of content downloads per month.                                        |
| `GenrePreference`          | Feature    | String    | User's preferred content genre.                                               |
| `UserRating`               | Feature    | Float     | User's rating of the service on a scale of 1 to 5.                            |
| `SupportTicketsPerMonth`   | Feature    | Integer   | Number of support tickets raised per month.                                   |
| `Gender`                   | Feature    | String    | Gender of the user: Male or Female.                                           |
| `WatchlistSize`            | Feature    | Float     | Number of items in the user's watchlist.                                      |
| `ParentalControl`          | Feature    | String    | Whether parental control is enabled: Yes or No.                               |
| `SubtitlesEnabled`         | Feature    | String    | Whether subtitles are enabled: Yes or No.                                     |
| `CustomerID`               | Identifier | String    | Unique identifier for each customer.                                          |
| `Churn`                    | Target     | Integer   | Indicates whether the customer has churned: 1 for churned, 0 for not churned. |

## Data Organization

The dataset contains three types of columns:

### Features

Features are the input variables used by the machine learning model to predict customer churn.

Examples include:

* Account age
* Monthly charges
* Subscription type
* Viewing behavior
* Payment method
* Support tickets
* User rating

### Identifier

`CustomerID` uniquely identifies each customer.

This column is expected to be excluded from model training because it identifies the customer rather than providing meaningful predictive information.

### Target

`Churn` is the target variable.

```text
1 = Churned
0 = Not Churned
```

## Data Preparation

Before model training, the dataset will be examined for:

* Missing values
* Duplicate records
* Incorrect data types
* Outliers
* Categorical variables
* Class imbalance
* Potentially irrelevant features
* Potential data leakage

The preprocessing steps and decisions will be documented in the project's notebooks and source code.

## Dataset License and Attribution

This dataset was obtained from Kaggle. Please refer to the original Kaggle dataset page for its applicable license, terms of use, and attribution requirements.

Original source:

[Predictive Analytics for Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset?utm_source=chatgpt.com)
