# Customer Retention Learning Agent

An AI-driven system that predicts customer churn, recommends personalized retention offers, and improves its strategy using customer feedback.

## Project Description

This project implements a customer retention learning agent built on top of a churn prediction model. The application analyzes a customer's profile, estimates churn probability, selects retention offers, captures customer feedback, and updates learned parameters over time.

Core components:

- Churn prediction using the trained TensorFlow model in `model.h5`
- Offer recommendation logic in `app.py`
- Feedback-driven learning through `logs/update_history.csv` and `logs/parameters.csv`
- Diagnostics and report generation through `report_visuals.py`

High-level workflow:

`Customer Data -> Predict Churn -> Generate Offer -> Collect Feedback -> Learn and Update`

## Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv312
   ```

3. Activate the virtual environment:

   Windows:

   ```powershell
   .\.venv312\Scripts\activate
   ```

   Linux / macOS:

   ```bash
   source .venv312/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## How to Run the Project

### Reset Learning State

Use this if you want to start fresh without previous feedback or learned parameters:

```powershell
.\.venv312\Scripts\python.exe reset_state.py
```

To also remove generated diagnostics:

```powershell
.\.venv312\Scripts\python.exe reset_state.py --clear-reports
```

### Run the Streamlit App

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py --server.headless true
```

Then open:

```text
http://localhost:8501
```

### Generate Reports

```powershell
.\.venv312\Scripts\python.exe report_visuals.py
```

This generates diagnostics in `reports/learning_diagnostics/`, including the HTML dashboard:

```text
reports/learning_diagnostics/index.html
```

## Example Input/Output

### Example Input

Customer ID:

```text
15634602
```

Example customer characteristics:

- Credit Score: 600
- Balance: 50000
- NumOfProducts: 1
- IsActiveMember: 0

### Example Output

- Churn Probability: `0.82`
- Risk Level: High
- Key Drivers: low engagement and low product usage
- Recommended Offer: cashback bonus to increase activity
- Selected Features: `IsActiveMember`, `NumOfProducts`
- Simulated Safe Outcome: churn reduced to a lower predicted value when a safe offer is available

### Example Feedback Loop

1. The customer accepts or rejects the offer.
2. The system stores the interaction in `logs/update_history.csv`.
3. Learned parameters are updated in `logs/parameters.csv`.
4. Future offer selection uses the updated state.

## Project Structure

```text
.
|-- app.py
|-- model.h5
|-- reset_state.py
|-- report_visuals.py
|-- logs/
|   |-- update_history.csv
|   `-- parameters.csv
|-- reports/
|   `-- learning_diagnostics/
`-- requirements.txt
```

## Notes

- The app may intentionally refuse to recommend an automated offer if all evaluated offers increase predicted churn.
- Resetting logs is enough for a fresh learning run; report files can be regenerated with `report_visuals.py`.
- Learning updates are based on recorded customer feedback and parameter snapshots.
