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
