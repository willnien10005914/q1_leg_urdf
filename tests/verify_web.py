"""Browser integration: offline URDF meshes, skill modes, seeking and video."""
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    root=Path(__file__).resolve().parents[1]
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8876/web/')
        page.wait_for_function("document.querySelector('#status').textContent.includes('已擷取')")
        page.locator('#play').click()
        page.locator('#timeline').evaluate("e=>{e.value='2';e.dispatchEvent(new Event('input',{bubbles:true}));}")
        assert page.locator('#time').inner_text()=='2.00 s'
        page.screenshot(path=str(root/'outputs/web_preview.png'),full_page=True)
        for skill in ('stand','walk','sidestep'):
            page.locator(f'[data-skill="{skill}"]').click()
            assert 'active' in page.locator(f'[data-skill="{skill}"]').get_attribute('class')
        page.locator('#source').select_option('policy')
        page.wait_for_timeout(500)
        assert 'PPO' in page.locator('#status').inner_text()
        page.locator('[data-skill="stand"]').click()
        page.wait_for_timeout(500)
        assert 'PPO' in page.locator('#mode-label').inner_text()
        response=page.request.get('http://127.0.0.1:8876/.git/config')
        assert response.status==404
        response=page.request.get('http://127.0.0.1:8876/checkpoints/stand/latest.zip')
        assert response.status==404
        page.locator('#source').select_option('reference')
        page.locator('[data-skill="sidestep"]').click()
        page.locator('#reference-video').evaluate('(v)=>v.load()')
        page.wait_for_function("document.querySelector('video').readyState>=1")
        assert page.locator('video').evaluate('(v)=>v.duration')>6.9
        # Resizing must enlarge the clip and resize WebGL without reloading video.
        before=page.locator('#reference-video').bounding_box()['width']
        divider=page.locator('#pane-resizer').bounding_box()
        page.mouse.move(divider['x']+divider['width']/2,divider['y']+150)
        page.mouse.down()
        page.mouse.move(divider['x']-240,divider['y']+150,steps=12)
        page.mouse.up()
        assert page.locator('#reference-video').bounding_box()['width']>before+200
        page.wait_for_function("Math.abs(document.querySelector('#canvas canvas').clientWidth-document.querySelector('#canvas').clientWidth)<2")
        page.locator('#pane-resizer').focus()
        page.keyboard.press('Home')
        page.locator('#expand-video').click()
        assert page.locator('#expand-video').get_attribute('aria-pressed')=='true'
        assert page.locator('#reference-video').bounding_box()['width']>700
        page.screenshot(path=str(root/'outputs/web_video_expanded.png'),full_page=True)
        page.locator('#expand-video').click()
        page.locator('[data-video-source="overlay"]').click()
        page.wait_for_function("document.querySelector('video').readyState>=1")
        assert 'pose_overlay' in page.locator('video').get_attribute('src')
        page.locator('[data-video-source="original"]').click()
        page.wait_for_function("document.querySelector('video').readyState>=1")
        for replay in ('v5_13115','upper_body'):
            page.locator('#source').select_option(replay)
            page.wait_for_function("document.querySelector('#status').textContent.includes('7.20 秒原生')")
            assert page.locator('#keyboard-toggle').is_disabled()
            assert page.locator('#sidestep-sequence').is_disabled()
            assert page.locator('[data-skill="walk"]').is_disabled()
            page.locator('#timeline').evaluate("e=>{e.value='3.8';e.dispatchEvent(new Event('input',{bubbles:true}));}")
            assert page.locator('#time').inner_text()=='3.80 s'
        page.locator('[data-video-source="upper_body"]').click()
        page.wait_for_function("document.querySelector('video').readyState>=1")
        assert page.locator('video').evaluate('v=>v.duration')==7.2
        assert '未重訓' in page.locator('#policy-version').inner_text()
        page.locator('#source').select_option('reference')
        page.reload()
        page.wait_for_function("document.querySelector('#status').textContent.includes('已擷取')")
        assert abs(page.locator('aside').bounding_box()['width']-420)<2
        page.set_viewport_size({'width':390,'height':844})
        assert not page.locator('#pane-resizer').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(root/'outputs/web_mobile.png'),full_page=True)
        assert not errors,errors
        browser.close()
    print('Web verified: skill modes, video resize/expand, keyboard, persistence, source switch, canvas resize and mobile.')

if __name__=='__main__':
    main()
