/**
 * useVoiceInput
 *
 * Manages the full microphone → ElevenLabs STT lifecycle:
 *   1. Requests mic permission via MediaRecorder API
 *   2. Collects audio chunks while recording
 *   3. On stop, POSTs the blob to ElevenLabs Scribe v2
 *   4. Returns the transcript via `onTranscript` callback
 *
 * States exposed:
 *   isRecording  — mic is actively capturing audio
 *   isTranscribing — audio uploaded, waiting for API response
 *   error        — last error message (string | null)
 *
 * Usage:
 *   const { isRecording, isTranscribing, error, startRecording, stopRecording } =
 *     useVoiceInput({ onTranscript: (text) => setInput(text) })
 */

import { useCallback, useRef, useState } from 'react'
import { transcribeAudio } from '../services/elevenlabs'

export function useVoiceInput({ onTranscript }) {
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [error, setError] = useState(null)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)

  const startRecording = useCallback(async () => {
    setError(null)

    // Guard: already recording
    if (isRecording) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Pick a MIME type the browser actually supports
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg']
        .find((m) => MediaRecorder.isTypeSupported(m)) ?? ''

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        // Stop all tracks so the browser mic indicator disappears
        streamRef.current?.getTracks().forEach((t) => t.stop())

        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })
        chunksRef.current = []

        if (blob.size === 0) {
          setError('No audio captured. Please try again.')
          setIsTranscribing(false)
          return
        }

        setIsTranscribing(true)
        try {
          const text = await transcribeAudio(blob)
          if (text) {
            onTranscript(text)
          } else {
            setError('No speech detected. Please try again.')
          }
        } catch (err) {
          setError(err.message ?? 'Transcription failed')
        } finally {
          setIsTranscribing(false)
        }
      }

      recorder.start()
      setIsRecording(true)
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Microphone permission denied. Please allow access and try again.')
      } else if (err.name === 'NotFoundError') {
        setError('No microphone found on this device.')
      } else {
        setError(err.message ?? 'Could not start recording')
      }
    }
  }, [isRecording, onTranscript])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }, [isRecording])

  const clearError = useCallback(() => setError(null), [])

  return {
    isRecording,
    isTranscribing,
    error,
    clearError,
    startRecording,
    stopRecording,
  }
}
