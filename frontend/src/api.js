const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Sentinel that the backend appends before the final JSON metadata chunk.
// Must match METADATA_SENTINEL in answer_service.py
const METADATA_SENTINEL = '\n[[METADATA]]'

async function parseError(response) {
  try {
    const data = await response.json()
    return data.detail || data.message || response.statusText
  } catch {
    return response.statusText || 'Request failed'
  }
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  return response.json()
}

export async function sendChatMessage(sessionId, message) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      stream: false,
    }),
  })

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  return response.json()
}

/**
 * Stream a chat message from the backend.
 *
 * The backend sends plain text tokens one by one, then finishes with:
 *   "\n[[METADATA]]" + JSON(ChatResponse)
 *
 * @param {string|null}  sessionId  - existing session or null for new session
 * @param {string}       message    - user's message text
 * @param {function}     onToken    - called with each text token as it arrives
 * @param {function}     onDone     - called once with the final ChatResponse metadata object
 * @returns {Promise<string>}       - resolves to the session_id
 */
export async function streamChatMessage(sessionId, message, onToken, onDone) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      stream: true,
    }),
  })

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  // Session ID arrives immediately in the header — no need to wait for stream end
  const newSessionId = response.headers.get('X-Session-Id') || sessionId

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')

  // Buffer accumulates chunks; sentinel may arrive split across multiple reads
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Check whether the terminal metadata sentinel has arrived
    const sentinelIdx = buffer.indexOf(METADATA_SENTINEL)
    if (sentinelIdx !== -1) {
      // Flush any text that arrived before the sentinel
      const trailingText = buffer.slice(0, sentinelIdx)
      if (trailingText) onToken(trailingText)

      // Parse and deliver the metadata JSON
      const jsonStr = buffer.slice(sentinelIdx + METADATA_SENTINEL.length)
      if (jsonStr.trim()) {
        try {
          onDone(JSON.parse(jsonStr))
        } catch {
          // Malformed JSON — the text answer is still visible; silently skip
        }
      }
      break
    }

    // Sentinel not yet seen — deliver the buffered text and reset
    onToken(buffer)
    buffer = ''
  }

  return newSessionId
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new Error('Backend unavailable')
  }
  return response.json()
}

