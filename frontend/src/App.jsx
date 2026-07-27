import { useCallback, useEffect, useRef, useState } from 'react'
import { checkHealth, sendChatMessage, uploadDocument } from './api'
import { useVoiceInput } from './hooks/useVoiceInput'
import { useVoiceOutput } from './hooks/useVoiceOutput'
import './App.css'

const SESSION_KEY = 'fin-rag-session-id'

// --------------------------------------------------------------------------
// Mic button — shown inside the chat form
// --------------------------------------------------------------------------
function MicButton({ isRecording, isTranscribing, onClick }) {
  let title = 'Start voice input'
  if (isRecording) title = 'Stop recording'
  if (isTranscribing) title = 'Transcribing…'

  return (
    <button
      type="button"
      className={[
        'btn-icon',
        'mic-btn',
        isRecording ? 'mic-btn--recording' : '',
        isTranscribing ? 'mic-btn--transcribing' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={onClick}
      disabled={isTranscribing}
      title={title}
      aria-label={title}
    >
      {isTranscribing ? (
        // Small inline spinner
        <span className="btn-spinner" aria-hidden="true" />
      ) : (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          width="18"
          height="18"
          aria-hidden="true"
        >
          <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm6.364 9.05a.75.75 0 0 1 .736.912A7.003 7.003 0 0 1 12.75 17.93V21h2.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1 0-1.5h2.5v-3.07a7.003 7.003 0 0 1-6.35-6.968.75.75 0 0 1 1.5 0 5.5 5.5 0 0 0 11 0 .75.75 0 0 1 .764-.912z" />
        </svg>
      )}
    </button>
  )
}

// --------------------------------------------------------------------------
// Speaker button — shown on every assistant bubble
// --------------------------------------------------------------------------
function SpeakerButton({ messageId, text, playingId, loadingId, onSpeak }) {
  const isLoading = loadingId === messageId
  const isPlaying = playingId === messageId

  let title = 'Read aloud'
  if (isLoading) title = 'Loading audio…'
  if (isPlaying) title = 'Stop'

  return (
    <button
      type="button"
      className={[
        'btn-icon',
        'speaker-btn',
        isPlaying ? 'speaker-btn--playing' : '',
        isLoading ? 'speaker-btn--loading' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onClick={() => onSpeak(messageId, text)}
      disabled={isLoading}
      title={title}
      aria-label={title}
    >
      {isLoading ? (
        <span className="btn-spinner" aria-hidden="true" />
      ) : isPlaying ? (
        // Stop square icon
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          width="15"
          height="15"
          aria-hidden="true"
        >
          <rect x="4" y="4" width="16" height="16" rx="2" />
        </svg>
      ) : (
        // Speaker icon
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          width="15"
          height="15"
          aria-hidden="true"
        >
          <path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 0 0 1.5 12c0 .898.121 1.768.35 2.595.341 1.24 1.518 1.905 2.659 1.905h1.93l4.5 4.5c.945.945 2.561.276 2.561-1.06V4.06zm5.084 1.046a.75.75 0 0 1 1.06 0c3.808 3.807 3.808 9.98 0 13.788a.75.75 0 0 1-1.06-1.06 8.25 8.25 0 0 0 0-11.668.75.75 0 0 1 0-1.06zm-1.82 3.81a.75.75 0 0 1 1.06 0 5.25 5.25 0 0 1 0 7.428.75.75 0 0 1-1.06-1.06 3.75 3.75 0 0 0 0-5.308.75.75 0 0 1 0-1.06z" />
        </svg>
      )}
    </button>
  )
}

// --------------------------------------------------------------------------
// Main App
// --------------------------------------------------------------------------
function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [backendOk, setBackendOk] = useState(null)

  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [uploadError, setUploadError] = useState('')

  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem(SESSION_KEY) || null,
  )
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [chatError, setChatError] = useState('')
  const chatEndRef = useRef(null)

  // Voice input hook — transcript lands in the text input
  const onTranscript = useCallback(
    (text) => setInput((prev) => (prev ? `${prev} ${text}` : text)),
    [], // setInput is stable (React state setter)
  )

  const {
    isRecording,
    isTranscribing,
    error: sttError,
    clearError: clearSttError,
    startRecording,
    stopRecording,
  } = useVoiceInput({ onTranscript })

  // Voice output hook — speaks assistant messages on demand
  const {
    playingId,
    loadingId,
    error: ttsError,
    clearError: clearTtsError,
    speak,
  } = useVoiceOutput()

  useEffect(() => {
    checkHealth()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false))
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, sending])

  function handleMicClick() {
    clearSttError()
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  async function handleUpload(event) {
    event.preventDefault()
    if (!selectedFile) return
    setUploading(true)
    setUploadError('')
    setUploadResult(null)
    try {
      const result = await uploadDocument(selectedFile)
      setUploadResult(result)
      setSelectedFile(null)
      event.target.reset()
    } catch (error) {
      setUploadError(error.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  async function handleSend(event) {
    event.preventDefault()
    const text = input.trim()
    if (!text || sending) return

    // Stop recording if the user hits Send mid-session
    if (isRecording) stopRecording()

    setSending(true)
    setChatError('')
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])

    try {
      const response = await sendChatMessage(sessionId, text)
      if (response.session_id) {
        setSessionId(response.session_id)
        localStorage.setItem(SESSION_KEY, response.session_id)
      }
      setMessages((prev) => [
        ...prev,
        {
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: response.answer,
          citations: response.citations || [],
          webSources: response.web_sources || [],
          confidence: response.confidence_score,
          usedWeb: response.used_web_fallback,
          faithfulness: response.faithfulness,
          answerRelevancy: response.answer_relevancy,
          precision: response.precision,
          recall: response.recall,
        },
      ])
    } catch (error) {
      setChatError(error.message || 'Failed to send message')
    } finally {
      setSending(false)
    }
  }

  function startNewSession() {
    setSessionId(null)
    localStorage.removeItem(SESSION_KEY)
    setMessages([])
    setChatError('')
  }

  // Combined voice error — show whichever fired last
  const voiceError = sttError || ttsError

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div>
            <h1>Financial Document RAG</h1>
            <p>Upload PDFs and ask questions grounded in your documents</p>
          </div>
          <div className={`status-pill ${backendOk ? 'online' : 'offline'}`}>
            {backendOk === null ? 'Checking…' : backendOk ? 'Backend online' : 'Backend offline'}
          </div>
        </div>
      </header>

      <main className="main">
        <nav className="tabs">
          <button
            type="button"
            className={activeTab === 'chat' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button
            type="button"
            className={activeTab === 'upload' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('upload')}
          >
            Upload
          </button>
        </nav>

        {/* ---------------------------------------------------------------- */}
        {/* Upload panel                                                      */}
        {/* ---------------------------------------------------------------- */}
        {activeTab === 'upload' && (
          <section className="panel">
            <h2>Upload Document</h2>
            <p className="hint">
              PDF only, up to 50 MB. Ingestion can take 1–3 minutes for large files — keep this tab open.
            </p>

            <form className="upload-form" onSubmit={handleUpload}>
              <label className="file-drop">
                <input
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  disabled={uploading}
                />
                <span>{selectedFile ? selectedFile.name : 'Choose a PDF file'}</span>
              </label>

              <button type="submit" className="btn primary" disabled={!selectedFile || uploading}>
                {uploading ? 'Processing…' : 'Upload & Index'}
              </button>
            </form>

            {uploading && (
              <div className="notice loading">
                <div className="spinner" />
                <p>Parsing, embedding, and indexing your document. This may take a few minutes.</p>
              </div>
            )}
            {uploadError && <div className="notice error">{uploadError}</div>}
            {uploadResult && (
              <div className="notice success">
                <strong>Document ready</strong>
                <ul>
                  <li>ID: {uploadResult.document_id}</li>
                  <li>File: {uploadResult.filename}</li>
                  <li>Pages: {uploadResult.num_pages}</li>
                  <li>Chunks: {uploadResult.num_chunks}</li>
                </ul>
              </div>
            )}
          </section>
        )}

        {/* ---------------------------------------------------------------- */}
        {/* Chat panel                                                        */}
        {/* ---------------------------------------------------------------- */}
        {activeTab === 'chat' && (
          <section className="panel chat-panel">
            <div className="chat-toolbar">
              <h2>Chat</h2>
              <button type="button" className="btn ghost" onClick={startNewSession}>
                New session
              </button>
            </div>

            <div className="messages">
              {messages.length === 0 && (
                <div className="empty-state">
                  Upload a PDF first, then ask questions like "What was the revenue?" or "Summarize key risks."
                </div>
              )}

              {messages.map((msg, index) => (
                <div key={msg.id ?? index} className={`message ${msg.role}`}>
                  <div className="bubble">
                    <p>{msg.content}</p>

                    {msg.role === 'assistant' && (
                      <>
                        {/* Citations */}
                        {msg.citations?.length > 0 && (
                          <div className="citations">
                            <span>Sources:</span>
                            {msg.citations.map((c, i) => (
                              <span key={i} className="citation-tag">
                                p.{c.page_number}
                                {c.section ? ` · ${c.section}` : ''}
                              </span>
                            ))}
                          </div>
                        )}

                        {/* Eval metrics */}
                        <div className="meta">
                          {typeof msg.confidence === 'number' && (
                            <span>Confidence: {(msg.confidence * 100).toFixed(0)}%</span>
                          )}
                          {typeof msg.faithfulness === 'number' && (
                            <span>Faithfulness: {msg.faithfulness.toFixed(2)}</span>
                          )}
                          {typeof msg.answerRelevancy === 'number' && (
                            <span>Relevancy: {msg.answerRelevancy.toFixed(2)}</span>
                          )}
                          {typeof msg.precision === 'number' && (
                            <span>Precision: {msg.precision.toFixed(2)}</span>
                          )}
                          {typeof msg.recall === 'number' && (
                            <span>Recall: {msg.recall.toFixed(2)}</span>
                          )}
                          {msg.usedWeb ? <span>· Web fallback used</span> : null}
                        </div>

                        {/* Speaker button */}
                        <div className="bubble-actions">
                          <SpeakerButton
                            messageId={msg.id ?? `idx-${index}`}
                            text={msg.content}
                            playingId={playingId}
                            loadingId={loadingId}
                            onSpeak={speak}
                          />
                        </div>
                      </>
                    )}
                  </div>
                </div>
              ))}

              {sending && (
                <div className="message assistant">
                  <div className="bubble typing">Thinking…</div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Voice / chat error banners */}
            {voiceError && (
              <div className="notice error notice--dismissible">
                <span>{voiceError}</span>
                <button
                  type="button"
                  className="notice-dismiss"
                  onClick={() => { clearSttError(); clearTtsError() }}
                  aria-label="Dismiss"
                >
                  ×
                </button>
              </div>
            )}
            {chatError && <div className="notice error">{chatError}</div>}

            {/* Recording status banner */}
            {isRecording && (
              <div className="notice notice--recording">
                <span className="recording-dot" aria-hidden="true" />
                Recording… tap the mic again to stop.
              </div>
            )}

            {/* Chat input form */}
            <form className="chat-form" onSubmit={handleSend}>
              <input
                type="text"
                placeholder={
                  isRecording
                    ? 'Listening…'
                    : isTranscribing
                      ? 'Transcribing…'
                      : 'Ask a question about your documents…'
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={sending}
                aria-label="Chat input"
              />

              <MicButton
                isRecording={isRecording}
                isTranscribing={isTranscribing}
                onClick={handleMicClick}
              />

              <button
                type="submit"
                className="btn primary"
                disabled={!input.trim() || sending}
              >
                Send
              </button>
            </form>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
