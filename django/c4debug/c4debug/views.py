import xml.etree.ElementTree as ET

import django.core.handlers.wsgi
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def replay(request: django.core.handlers.wsgi.WSGIRequest):
    debug_file = request.FILES['debug_file']
    text = []
    for line in debug_file:
        line = line.decode('utf-8')
        text.append(line)
    root = ET.fromstringlist(text)
    moves = list(root)

    # return HttpResponse(f'<html><body>Request: {root.tag}</body></html>')
    return HttpResponse(root.tag)
