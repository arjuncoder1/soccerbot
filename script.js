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

// ---------- hero scripted sequence: messi_bot.mp4 (first 3s) -> messi_bot_successful (looped) ----------
const HERO_CLIP_SECONDS = 3;
const HERO_SECOND_CLIP_SECONDS = 5;

const heroVideo = document.getElementById('heroVideo');
const heroSecondClip = document.getElementById('heroSecondClip');
const segClip = document.getElementById('segClip');
const segReel = document.getElementById('segReel');
const heroSkip = document.getElementById('heroSkip');

let switchedToSecondClip = false;

function switchToSecondClip() {
  if (switchedToSecondClip) return;
  switchedToSecondClip = true;
  heroVideo.pause();
  heroVideo.classList.remove('active');
  segClip.classList.add('done');

  heroSecondClip.currentTime = 0;
  heroSecondClip.classList.add('active');
  heroSecondClip.play().catch(() => {});
  segReel.querySelector('i').style.width = '100%';
  heroSkip.style.display = 'none';
}

heroVideo.addEventListener('timeupdate', () => {
  if (switchedToSecondClip) return;
  const pct = Math.min(100, (heroVideo.currentTime / HERO_CLIP_SECONDS) * 100);
  segClip.querySelector('i').style.width = pct + '%';
  if (heroVideo.currentTime >= HERO_CLIP_SECONDS) switchToSecondClip();
});

heroVideo.addEventListener('ended', switchToSecondClip);
heroSkip.addEventListener('click', switchToSecondClip);

heroSecondClip.addEventListener('timeupdate', () => {
  if (heroSecondClip.currentTime >= HERO_SECOND_CLIP_SECONDS) heroSecondClip.currentTime = 0;
});

// autoplay can be blocked on some browsers until interaction; fall back gracefully
heroVideo.play().catch(() => {
  // if autoplay is blocked, go straight to the second clip on first user interaction
  const kick = () => { heroVideo.play().catch(() => switchToSecondClip()); document.removeEventListener('click', kick); };
  document.addEventListener('click', kick, { once: true });
});
