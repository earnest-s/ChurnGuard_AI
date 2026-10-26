import { useState } from 'react'

const rawApiUrl = (import.meta.env.VITE_API_URL || '').trim()
const API_URL =
  typeof window !== 'undefined' && window.location.protocol === 'https:'
    ? rawApiUrl.replace(/^http:\/\//i, 'https://')
    : rawApiUrl

const initialFormState = {
  tenure: '',
  monthly_charges: '',
  contract_type: '0',
  support_calls: '',
  internet_service: '',
  payment_method: '',
  paperless_billing: '',
  online_security: '',
  tech_support: '',
  streaming_tv: '',
  streaming_movies: '',
  partner: '',
  dependents: '',
  senior_citizen: '',
}

const YES_NO = ['Yes', 'No']
const SERVICE_OPTIONS = ['Yes', 'No', 'No internet service']

// Extra Telco fields the API accepts. Anything left blank falls back to the
// model's reference customer, and the response says which values were assumed.
const ADVANCED_FIELDS = [
  { name: 'internet_service', label: 'Internet Service', options: ['DSL', 'Fiber optic', 'No'] },
  {
    name: 'payment_method',
    label: 'Payment Method',
    options: [
      'Bank transfer (automatic)',
      'Credit card (automatic)',
      'Electronic check',
      'Mailed check',
    ],
  },
  { name: 'paperless_billing', label: 'Paperless Billing', options: YES_NO },
  { name: 'online_security', label: 'Online Security', options: SERVICE_OPTIONS },
  { name: 'tech_support', label: 'Tech Support', options: SERVICE_OPTIONS },
  { name: 'streaming_tv', label: 'Streaming TV', options: SERVICE_OPTIONS },
  { name: 'streaming_movies', label: 'Streaming Movies', options: SERVICE_OPTIONS },
  { name: 'partner', label: 'Partner', options: YES_NO },
  { name: 'dependents', label: 'Dependents', options: YES_NO },
  { name: 'senior_citizen', label: 'Senior Citizen', options: [['true', 'Yes'], ['false', 'No']] },
]

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

  const parseApiResponse = async (response) => {
    const text = await response.text()
    let data

    try {
      data = text ? JSON.parse(text) : {}
    } catch {
      throw new Error(`Invalid JSON response: ${text}`)
    }

    if (!response.ok) {
      throw new Error(data.error || data.message || 'API request failed')
    }

    return data
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

      ADVANCED_FIELDS.forEach(({ name }) => {
        const value = formData[name]
        if (value === '') return
        payload[name] = name === 'senior_citizen' ? value === 'true' : value
      })

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await parseApiResponse(response)
      const probability = data.churn_probability
      const label = data.prediction
      const importance = data.feature_importance

      if (typeof probability !== 'number' || typeof label !== 'string') {
        throw new Error('Invalid prediction response schema')
      }

      setResult({
        probability,
        label,
        importance,
        explanation: typeof data.explanation === 'string' ? data.explanation : '',
        modelName: typeof data.model_name === 'string' ? data.model_name : '',
        xaiMethod: typeof data.xai_method === 'string' ? data.xai_method : '',
      })
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
      const response = await fetch(`${API_URL}/drift-status`)
      const data = await parseApiResponse(response)
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
      const response = await fetch(`${API_URL}/retrain`, {
        method: 'POST',
      })

      const data = await parseApiResponse(response)
      setRetrainMessage(data.status)
      setDriftStatus({ drift_detected: false, drift_count: 0 })
    } catch (err) {
      setRetrainError(err.message || 'Could not retrain model.')
    } finally {
      setRetrainLoading(false)
    }
  }

  const resultClass = result?.label === 'High Risk' ? 'result high-risk' : 'result low-risk'

  const featureEntries = result?.importance
    ? Object.entries(result.importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
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
            <span>Support Calls (legacy model only)</span>
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
              <h2>{result.label}</h2>
              <p>
                Churn Probability: <strong>{(result.probability * 100).toFixed(2)}%</strong>
              </p>
            </article>

            <section className="insights">
              <h3>Model Insights{result.xaiMethod === 'shap' ? ' (SHAP)' : ''}</h3>
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

              {result.explanation && <p className="explanation">{result.explanation}</p>}
              {result.modelName && (
                <p className="model-meta">
                  Model: <strong>{result.modelName}</strong>
                  {result.xaiMethod ? ` · explanation: ${result.xaiMethod}` : ''}
                </p>
              )}
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
