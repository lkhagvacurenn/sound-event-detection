console.log("APP VERSION 4 LOADED");

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

const soundClass = document.getElementById('soundClass');
const confidenceText = document.getElementById('confidence');
const statusText = document.getElementById('status');
const alertBox = document.getElementById('alertBox');

const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');

canvas.width = 850;
canvas.height = 220;

let stream;
let analyser;
let animationId;
let audioContext;
let isRecording = false;
let isListening = false;
let lastAlertTime = 0;

const CONFIDENCE_THRESHOLD = 0.60;
const ALERT_COOLDOWN = 10000; // 10 seconds

const suspiciousSounds = [
  'speech',
  'walk',
  'door',
  'dog'
];

async function startListening() {

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1
    }
  });

  statusText.innerText = 'Listening...';
  statusText.style.background = '#16a34a';

  setupWaveform(stream);

  isListening = true;

  setTimeout(() => {
    predictLoop();
  }, 500);
}

async function predictLoop() {
  if (!isListening) return;
  await recordAndPredict();
  if (isListening) {
    predictLoop();
  }
}

function stopListening() {

  isListening = false;

  if(stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  if(animationId) {
    cancelAnimationFrame(animationId);
  }

  if(audioContext) {
    audioContext.close();
  }

  isRecording = false;

  statusText.innerText = 'Microphone Stopped';
  statusText.style.background = '#dc2626';
}

function setupWaveform(stream) {

  audioContext = new AudioContext();

  const source = audioContext.createMediaStreamSource(stream);

  analyser = audioContext.createAnalyser();

  analyser.fftSize = 2048;

  source.connect(analyser);

  drawWaveform();
}

function drawWaveform() {

  if(!analyser) {
    return;
  }

  const bufferLength = analyser.fftSize;

  const dataArray = new Uint8Array(bufferLength);

  analyser.getByteTimeDomainData(dataArray);

  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#3b82f6';

  ctx.beginPath();

  const sliceWidth = canvas.width / bufferLength;

  let x = 0;

  for(let i = 0; i < bufferLength; i++) {

    const v = dataArray[i] / 128.0;

    const y = v * canvas.height / 2;

    if(i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  ctx.lineTo(canvas.width, canvas.height / 2);

  ctx.stroke();

  animationId = requestAnimationFrame(drawWaveform);
}

function recordAndPredict() {
  return new Promise((resolve) => {

    if(isRecording || !stream) {
      resolve();
      return;
    }

    isRecording = true;

    const recorder = RecordRTC(stream, {
      type: 'audio',
      mimeType: 'audio/wav',
      recorderType: StereoAudioRecorder,
      desiredSampRate: 22050,
      numberOfAudioChannels: 1
    });

    recorder.startRecording();

    setTimeout(() => {

      recorder.stopRecording(async () => {

        const blob = recorder.getBlob();

        console.log("Blob size:", blob.size);

        const formData = new FormData();

        formData.append(
          'audio',
          blob,
          'recording.wav'
        );

        try {

          const response = await fetch(
            'http://127.0.0.1:5000/predict',
            {
              method: 'POST',
              body: formData
            }
          );

          const result = await response.json();

          console.log('FULL BACKEND RESULT:', result);
          console.log('Prediction:', result.class);
          console.log('Confidence:', result.confidence);
          console.log('All predictions:', result.all_predictions);

          if(result.error) {
            console.error(result.error);
            return;
          }

          if(result.confidence < CONFIDENCE_THRESHOLD) {

            soundClass.innerText = 'uncertain';

            confidenceText.innerText =
              (result.confidence * 100).toFixed(2) + '%';

            alertBox.style.display = 'none';

            return;
          }

          soundClass.innerText = result.class;

          confidenceText.innerText =
            (result.confidence * 100).toFixed(2) + '%';

          if(suspiciousSounds.includes(result.class)) {

            alertBox.style.display = 'block';

            const now = Date.now();

            if(now - lastAlertTime > ALERT_COOLDOWN) {

              sendDiscordWebhook(result.class);

              lastAlertTime = now;
            }

          } else {

            alertBox.style.display = 'none';
          }

        } catch(error) {

          console.error(error);

        } finally {

          isRecording = false;
          resolve();
        }

      });

    }, 4000);
  });
}

async function sendDiscordWebhook(sound) {

  const WEBHOOK_URL = 'YOUR_DISCORD_WEBHOOK_URL';

  if(WEBHOOK_URL === 'YOUR_DISCORD_WEBHOOK_URL') {
    return;
  }

  try {

    await fetch(WEBHOOK_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        content: `⚠ Suspicious sound detected: ${sound}`
      })
    });

  } catch(error) {

    console.error(error);
  }
}

startBtn.addEventListener('click', startListening);
stopBtn.addEventListener('click', stopListening);