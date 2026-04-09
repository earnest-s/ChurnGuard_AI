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
  const [driftStatus, setDriftStatus] = useState(null)
  const [driftLoading, setDriftLoading] = useState(false)
  const [driftError, setDriftError] = useState('')
  const [retrainLoading, setRetrainLoading] = useState(false)
  const [retrainMessage, setRetrainMessage] = useState('')
  const [retrainError, setRetrainError] = useState('')

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

  const handleCheckDrift = async () => {
    setDriftLoading(true)
    setDriftError('')

    try {
      const response = await fetch('http://localhost:8000/drift-status')
      if (!response.ok) {
        throw new Error('Failed to fetch drift status.')
      }

      const data = await response.json()
      setDriftStatus(data)
    } catch (err) {
      setDriftError(err.message || 'Could not load drift status.')
    } finally {
      setDriftLoading(false)
    }
  }

  const handleRetrain = async () => {
    setRetrainLoading(true)
    setRetrainMessage('')
    setRetrainError('')

    try {
      const response = await fetch('http://localhost:8000/retrain', {
        method: 'POST',
      })

      if (!response.ok) {
        const details = await response.text()
        throw new Error(details || 'Retraining failed.')
      }

      const data = await response.json()
      setRetrainMessage(data.status)
      setDriftStatus({ drift_detected: false, drift_count: 0 })
    } catch (err) {
      setRetrainError(err.message || 'Could not retrain model.')
    } finally {
      setRetrainLoading(false)
    }
  }

  const resultClass =
    result?.prediction === 'High Risk' ? 'result high-risk' : 'result low-risk'

  const featureEntries = result?.feature_importance
    ? Object.entries(result.feature_importance)
    : []

  const topFeature = featureEntries.reduce(
    (max, current) => (current[1] > max[1] ? current : max),
    ['', -Infinity],
  )

  const toLabel = (name) =>
    name
      .split('_')
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ')

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
          <>
            <article className={resultClass}>
              <h2>{result.prediction}</h2>
              <p>
                Churn Probability: <strong>{(result.churn_probability * 100).toFixed(2)}%</strong>
              </p>
            </article>

            <section className="insights">
              <h3>Model Insights</h3>
              <p className="insight-subtitle">
                Most influential feature: <strong>{toLabel(topFeature[0])}</strong>
              </p>

              <div className="importance-list">
                {featureEntries.map(([feature, value]) => {
                  const isTopFeature = feature === topFeature[0]

                  return (
                    <div
                      className={`importance-item ${isTopFeature ? 'top-feature' : ''}`}
                      key={feature}
                    >
                      <div className="importance-header">
                        <span>{toLabel(feature)}</span>
                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="importance-track">
                        <div className="importance-fill" style={{ width: `${value * 100}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>
          </>
        )}

        <section className="ops-panel">
          <h3>Monitoring</h3>
          <div className="ops-actions">
            <button type="button" onClick={handleCheckDrift} disabled={driftLoading}>
              {driftLoading ? 'Checking...' : 'Check Drift Status'}
            </button>
            <button type="button" onClick={handleRetrain} disabled={retrainLoading}>
              {retrainLoading ? 'Retraining...' : 'Retrain Model'}
            </button>
          </div>

          {driftStatus && (
            <p className="ops-info">
              Drift Detected: <strong>{driftStatus.drift_detected ? 'Yes' : 'No'}</strong> | Drift
              Count: <strong>{driftStatus.drift_count}</strong>
            </p>
          )}

          {driftError && <p className="error">{driftError}</p>}
          {retrainError && <p className="error">{retrainError}</p>}
          {retrainMessage && <p className="success-message">{retrainMessage}</p>}
        </section>
      </section>
    </main>
  )
}
