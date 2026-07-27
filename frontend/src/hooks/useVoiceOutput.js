/**
 * useVoiceOutput
 *
 * Manages the ElevenLabs TTS → browser Audio playback lifecycle.
 *
 * One audio element is reused across calls; starting a new utterance
 * automatically stops any currently playing audio and revokes the old
 * blob URL to prevent memory leaks.
 *
 * States exposed:
 *   playingId    — the message ID currently being spoken (string | null)
 *   loadingId    — the message ID whose audio is still being fetched (string | null)
 *   error        — last error message (string | null)
 *
 * Usage:
 *   const { playingId, loadingId, error, speak, stop } = useVoiceOutput()
 *
 *   // In a message component:
 *   <button onClick={() => speak(msg.id, msg.content)}>
 *     {playingId === msg.id ? '⏹' : '🔊'}
 *   </button>
 */

import { useCallback, useRef, useState } from 'react'
import { textToSpeech } from '../services/elevenlabs'

export function useVoiceOutput() {
  const [playingId, setPlayingId] = useState(null)
  const [loadingId, setLoadingId] = useState(null)
  const [error, setError] = useState(null)

  const audioRef = useRef(null)
  const blobUrlRef = useRef(null)
  // Mirror state in refs so `speak` and `stop` never need to be recreated.
  const playingIdRef = useRef(null)
  const loadingIdRef = useRef(null)

  // Keep refs in sync with state.
  const setPlaying = useCallback((id) => {
    playingIdRef.current = id
    setPlayingId(id)
  }, [])
  const setLoading = useCallback((id) => {
    loadingIdRef.current = id
    setLoadingId(id)
  }, [])

  /** Stop any currently playing / loading audio. */
  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      // Detach onerror BEFORE clearing src so that setting src='' does not
      // fire the error handler and produce a spurious "Audio playback failed".
      audioRef.current.onerror = null
      audioRef.current.onended = null
      audioRef.current.src = ''
      audioRef.current = null
    }
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current)
      blobUrlRef.current = null
    }
    setPlaying(null)
    setLoading(null)
  }, [setPlaying, setLoading])

  /**
   * Fetch TTS audio for `text` and play it, tagged with `id` so the UI
   * knows which message is active.
   *
   * Calling speak() while the same id is already playing stops it (toggle).
   *
   * `speak` has an empty dependency array — it reads live values through refs
   * so its identity never changes, preventing re-render cascades in the parent.
   */
  const speak = useCallback(
    async (id, text) => {
      setError(null)

      // Toggle off if this message is already playing or loading
      if (playingIdRef.current === id || loadingIdRef.current === id) {
        stop()
        return
      }

      // Stop anything else that's running
      stop()

      setLoading(id)

      try {
        const blobUrl = await textToSpeech(text)
        blobUrlRef.current = blobUrl

        const audio = new Audio(blobUrl)
        audioRef.current = audio

        audio.onended = () => {
          setPlaying(null)
          URL.revokeObjectURL(blobUrl)
          blobUrlRef.current = null
        }

        audio.onerror = () => {
          setPlaying(null)
          setError('Audio playback failed')
          URL.revokeObjectURL(blobUrl)
          blobUrlRef.current = null
        }

        await audio.play()
        setLoading(null)
        setPlaying(id)
      } catch (err) {
        setLoading(null)
        setPlaying(null)
        setError(err.message ?? 'Text-to-speech failed')
      }
    },
    // Empty deps — all mutable values are accessed via refs or stable setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const clearError = useCallback(() => setError(null), [])

  return {
    playingId,
    loadingId,
    error,
    clearError,
    speak,
    stop,
  }
}
