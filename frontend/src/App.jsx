import { useState } from 'react'

const initialFormState = {
  tenure: '',
  monthly_charges: '',
  contract_type: '0',
  support_calls: '',
}

export default function App() {
  const [formData, setFormData] = useState(initialFormState)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const handleChange = (event) => {
    const { name, value } = event.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)

    try {
      const payload = {
        tenure: Number(formData.tenure),
        monthly_charges: Number(formData.monthly_charges),
        contract_type: Number(formData.contract_type),
        support_calls: Number(formData.support_calls),
      }

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const details = await response.text()
        throw new Error(details || 'Prediction request failed.')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong while predicting churn.')
    } finally {
      setLoading(false)
    }
  }

  const resultClass =
    result?.prediction === 'High Risk' ? 'result high-risk' : 'result low-risk'

  return (
    <main className="page-bg">
      <section className="card">
        <header className="card-header">
          <p className="eyebrow">Customer Retention</p>
          <h1>Churn Risk Predictor</h1>
          <p className="subtitle">Estimate whether a customer is likely to churn.</p>
        </header>

        <form className="form-grid" onSubmit={handleSubmit}>
          <label>
            <span>Tenure (months)</span>
            <input
              type="number"
              name="tenure"
              min="0"
              step="1"
              required
              value={formData.tenure}
              onChange={handleChange}
            />
          </label>

          <label>
            <span>Monthly Charges</span>
            <input
              type="number"
              name="monthly_charges"
              min="0"
              step="0.01"
              required
              value={formData.monthly_charges}
              onChange={handleChange}
            />
          </label>

          <label>
            <span>Contract Type</span>
            <select
              name="contract_type"
              value={formData.contract_type}
              onChange={handleChange}
            >
              <option value="0">0 - Month-to-Month</option>
              <option value="1">1 - One Year</option>
              <option value="2">2 - Two Year</option>
            </select>
          </label>

          <label>
            <span>Support Calls</span>
            <input
              type="number"
              name="support_calls"
              min="0"
              step="1"
              required
              value={formData.support_calls}
              onChange={handleChange}
            />
          </label>

          <button type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Churn'}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <article className={resultClass}>
            <h2>{result.prediction}</h2>
            <p>
              Churn Probability: <strong>{(result.churn_probability * 100).toFixed(2)}%</strong>
            </p>
          </article>
        )}
      </section>
    </main>
  )
}
