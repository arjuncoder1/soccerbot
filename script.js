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

// ---------- hero video: autoplay, loop just the first few seconds ----------
const HERO_CLIP_SECONDS = 4;
const heroVideo = document.getElementById('heroVideo');
heroVideo.addEventListener('timeupdate', () => {
  if (heroVideo.currentTime >= HERO_CLIP_SECONDS) heroVideo.currentTime = 0;
});
heroVideo.play().catch(() => {
  const kick = () => { heroVideo.play().catch(() => {}); document.removeEventListener('click', kick); };
  document.addEventListener('click', kick, { once: true });
});
