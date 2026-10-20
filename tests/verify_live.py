"""Exercise real PhysX inference and its Web controls, not recorded playback."""
from pathlib import Path
import json
import time
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from playwright.sync_api import sync_playwright

BASE='http://127.0.0.1:8876'
def get():
    with urlopen(BASE+'/api/live',timeout=3) as r:return json.load(r)
def command(skill,action):
    with urlopen(Request(BASE+'/api/live/command',data=json.dumps({'skill':skill,'action':action}).encode(),headers={'Content-Type':'application/json'}),timeout=3) as r:return json.load(r)
def wait_for(predicate,timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            state=get()
            if predicate(state):return state
        except (OSError,ValueError):pass
        time.sleep(.1)
    raise AssertionError('Live simulation did not reach the expected state')

def main():
    state=wait_for(lambda s:s.get('available',{}).get('stand'))
    assert state['available']['stand']
    command('stand','restart')
    state=wait_for(lambda s:s.get('status')=='running' and (s.get('frame') or {}).get('time',0)>0)
    first=state['frame']['time']
    assert first>0 and state['status']=='running'
    command('stand','pause')
    paused=wait_for(lambda s:s['status']=='paused')['frame']['time']
    time.sleep(.25)
    assert get()['frame']['time']==paused
    command('stand','play')
    time.sleep(.3)
    assert get()['frame']['time']>paused
    for skill,ready in get()['available'].items():
        if not ready:
            try:command(skill,'restart')
            except HTTPError as e:assert e.code==409
            else:raise AssertionError('Unpassed skill was allowed')
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(BASE+'/web/')
        page.wait_for_function("document.querySelector('#status').textContent.includes('已擷取')")
        page.locator('[data-skill="stand"]').click()
        page.locator('#source').select_option('live')
        page.wait_for_function("document.querySelector('#status').textContent.includes('執行中')")
        assert page.locator('#timeline').is_disabled()
        page.locator('#play').click()
        page.wait_for_function("document.querySelector('#status').textContent.includes('已暫停')")
        page.screenshot(path=str(Path(__file__).resolve().parents[1]/'outputs/web_live.png'),full_page=True)
        assert not errors,errors
        browser.close()
    print('Live PhysX verified: checkpoint gate, advancing state, pause/resume, rejected unpassed skills, browser controls.')
if __name__=='__main__':main()
