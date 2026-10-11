"""Keep the original deployment's plastic URL working."""
from ui.auditors import render_auditor
from ui.settings import settings

render_auditor("plastic", settings())
