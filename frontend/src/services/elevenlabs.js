/**
 * ElevenLabs API service
 *
 * STT: POST /v1/speech-to-text  (Scribe v2 model, multipart)
 * TTS: POST /v1/text-to-speech/{voiceId}  (returns raw MP3 bytes)
 *
 * Both functions throw an Error with a human-readable message on failure.
 *
 * Network notes:
 * - Each call gets its own AbortController with a 15 s timeout so a stale
 *   or rate-limited connection does not hang the UI indefinitely.
 * - "Failed to fetch" on the second call is typically either a timed-out
 *   CORS preflight (browser cached a stale Allow header) or a rate-limited
 *   API key. The timeout + explicit error mapping surfaces the real cause
 *   instead of the generic browser error string.
 */

const BASE_URL = 'https://api.elevenlabs.io'
const API_KEY = import.meta.env.VITE_ELEVENLABS_API_KEY
const VOICE_ID = import.meta.env.VITE_ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb'

// How long to wait for ElevenLabs to respond before giving up.
// STT can take a few seconds for long recordings; TTS is typically <2 s.
const REQUEST_TIMEOUT_MS = 15_000

/**
 * Create a fetch that automatically aborts after REQUEST_TIMEOUT_MS.
 * Returns { response, cleanup } — call cleanup() to clear the timeout
 * once the response is received (prevents a dangling timer leak).
 */
async function fetchWithTimeout(url, options) {
  const controller = new AbortController()
  const timerId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    const response = await fetch(url, { ...options, signal: controller.signal })
    return response
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('ElevenLabs request timed out — check your network or API key quota')
    }
    // "Failed to fetch" is a generic browser message for network errors
    // (CORS block, DNS failure, connection refused). Append the original
    // message so the developer can see it in the error banner.
    throw new Error(`Network error: ${err.message}`)
  } finally {
    // Always clear the timer to avoid a memory leak, regardless of outcome.
    clearTimeout(timerId)
  }
}

// --------------------------------------------------------------------------
// Speech-to-Text (Scribe v2)
// --------------------------------------------------------------------------

/**
 * Transcribe an audio Blob to text using the ElevenLabs Scribe v2 model.
 *
 * @param {Blob} audioBlob  — the recorded audio (webm / ogg / mp4 etc.)
 * @returns {Promise<string>}  — the transcribed text
 */
export async function transcribeAudio(audioBlob) {
  if (!API_KEY) {
    throw new Error('VITE_ELEVENLABS_API_KEY is not set')
  }

  const formData = new FormData()
  // ElevenLabs requires a filename with an extension so it can detect the format
  formData.append('file', audioBlob, 'recording.webm')
  formData.append('model_id', 'scribe_v2')

  const response = await fetchWithTimeout(`${BASE_URL}/v1/speech-to-text`, {
    method: 'POST',
    headers: {
      'xi-api-key': API_KEY,
    },
    body: formData,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText)
    throw new Error(`STT failed (${response.status}): ${errorText}`)
  }

  const data = await response.json()
  // API returns { text: "..." } at the top level for single-channel transcription
  return (data.text ?? '').trim()
}

// --------------------------------------------------------------------------
// Text-to-Speech
// --------------------------------------------------------------------------

/**
 * Convert text to speech and return a playable audio URL (object URL).
 *
 * Callers are responsible for calling URL.revokeObjectURL() when done
 * to avoid memory leaks.
 *
 * @param {string} text       — the text to speak
 * @param {string} [voiceId]  — optional override; defaults to VITE_ELEVENLABS_VOICE_ID
 * @returns {Promise<string>}  — a blob:// URL pointing at the MP3 audio
 */
export async function textToSpeech(text, voiceId = VOICE_ID) {
  if (!API_KEY) {
    throw new Error('VITE_ELEVENLABS_API_KEY is not set')
  }
  if (!text?.trim()) {
    throw new Error('No text provided for TTS')
  }

  const response = await fetchWithTimeout(
    `${BASE_URL}/v1/text-to-speech/${voiceId}?output_format=mp3_44100_128`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      }),
    },
  )

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText)
    throw new Error(`TTS failed (${response.status}): ${errorText}`)
  }

  const audioBlob = await response.blob()
  return URL.createObjectURL(audioBlob)
}
