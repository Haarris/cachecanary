// Copy the install command. Falls back to selecting the text if the clipboard API is unavailable or blocked.
(function () {
  var btn = document.getElementById("copy");
  var text = document.getElementById("install-cmd");
  if (!btn || !text) return;
  function flash(label) {
    btn.textContent = label;
    setTimeout(function () { btn.textContent = "Copy"; }, 1500);
  }
  function selectText() {
    var range = document.createRange();
    range.selectNodeContents(text);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    flash("Selected");
  }
  btn.addEventListener("click", function () {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text.textContent).then(function () { flash("Copied"); }, selectText);
    } else {
      selectText();
    }
  });
})();

// Guides menu: a <details> element, so it works without JavaScript. This only closes it on an
// outside click, on Escape, or after picking a link on the same page.
(function () {
  var menu = document.querySelector("details.guides");
  if (!menu) return;
  document.addEventListener("click", function (e) {
    if (menu.open && !menu.contains(e.target)) menu.open = false;
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && menu.open) {
      menu.open = false;
      menu.querySelector("summary").focus();
    }
  });
  menu.querySelectorAll(".menu a").forEach(function (a) {
    a.addEventListener("click", function () { menu.open = false; });
  });
})();
