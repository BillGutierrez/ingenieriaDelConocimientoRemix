(() => {
  const trigger = document.querySelector(".easter-egg-trigger");
  const dialog = document.querySelector("#love-note");

  if (!trigger || !dialog || typeof dialog.showModal !== "function") {
    return;
  }

  const clicksNeeded = Number(trigger.dataset.clicksNeeded);
  let clickCount = 0;
  let resetTimer;

  trigger.addEventListener("click", () => {
    clickCount += 1;
    window.clearTimeout(resetTimer);

    if (clickCount >= clicksNeeded) {
      clickCount = 0;
      dialog.showModal();
      return;
    }

    resetTimer = window.setTimeout(() => {
      clickCount = 0;
    }, 3500);
  });
})();
