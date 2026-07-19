// ---------- nav shadow on scroll ----------
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 20);
}, { passive: true });

// ---------- reveal-on-scroll ----------
const revealEls = document.querySelectorAll('.reveal');
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('in'); });
}, { threshold: 0.15 });
revealEls.forEach(el => io.observe(el));

// ---------- hero scripted sequence: messi_bot.mp4 (first 3s) -> YouTube reel ----------
const HERO_CLIP_SECONDS = 3;
const YT_ID = 'B6rOcKDwPHQ';

const heroVideo = document.getElementById('heroVideo');
const heroYtWrap = document.getElementById('heroYtWrap');
const heroYt = document.getElementById('heroYt');
const segClip = document.getElementById('segClip');
const segReel = document.getElementById('segReel');
const heroSkip = document.getElementById('heroSkip');

let switchedToReel = false;

function switchToReel() {
  if (switchedToReel) return;
  switchedToReel = true;
  heroVideo.pause();
  heroVideo.classList.remove('active');
  segClip.classList.add('done');

  const src = `https://www.youtube-nocookie.com/embed/${YT_ID}?autoplay=1&mute=1&controls=1&rel=0&playsinline=1`;
  heroYt.src = src;
  heroYtWrap.classList.add('active');
  segReel.querySelector('i').style.width = '100%';
  heroSkip.style.display = 'none';
}

heroVideo.addEventListener('timeupdate', () => {
  if (switchedToReel) return;
  const pct = Math.min(100, (heroVideo.currentTime / HERO_CLIP_SECONDS) * 100);
  segClip.querySelector('i').style.width = pct + '%';
  if (heroVideo.currentTime >= HERO_CLIP_SECONDS) switchToReel();
});

heroVideo.addEventListener('ended', switchToReel);
heroSkip.addEventListener('click', switchToReel);

// autoplay can be blocked on some browsers until interaction; fall back gracefully
heroVideo.play().catch(() => {
  // if autoplay is blocked, go straight to the reel on first user interaction
  const kick = () => { heroVideo.play().catch(() => switchToReel()); document.removeEventListener('click', kick); };
  document.addEventListener('click', kick, { once: true });
});
