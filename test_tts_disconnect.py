import asyncio
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.modules.setdefault('sounddevice',SimpleNamespace())
sys.modules.setdefault('websockets',SimpleNamespace(ConnectionClosed=type('ConnectionClosed',(Exception,),{})))
from elevenlabs_tts import ElevenLabsTTS

class Socket:
    def __init__(self,messages=(),error=None):self.messages=messages;self.error=error
    def __aiter__(self):
        async def events():
            for message in self.messages:yield json.dumps(message)
            if self.error:raise self.error
        return events()

class TTSDisconnectTest(unittest.IsolatedAsyncioTestCase):
    def make_tts(self,socket):
        speaker=SimpleNamespace(reset=lambda:None,finished=False)
        speaker.mark_end=lambda:setattr(speaker,'finished',True)
        with patch('elevenlabs_tts.Speaker',return_value=speaker):
            tts=ElevenLabsTTS('unused','voice','model')
        tts.start('reply');tts._opened=True;tts._ws=socket
        return tts
    async def test_clean_disconnect_unblocks_wait_with_error(self):
        tts=self.make_tts(Socket())
        await tts._read()
        with self.assertRaisesRegex(RuntimeError,'closed'):
            await asyncio.wait_for(tts.wait_until_spoken(),timeout=0.1)
    async def test_reader_error_reaches_playback_waiter(self):
        tts=self.make_tts(Socket(error=ValueError('bad audio message')))
        await tts._read()
        with self.assertRaisesRegex(ValueError,'bad audio message'):
            await asyncio.wait_for(tts.wait_until_spoken(),timeout=0.1)
    async def test_service_error_unblocks_wait_with_diagnostic(self):
        tts=self.make_tts(Socket([{'error':'quota_exceeded','message':'No credits'}]))
        await tts._read()
        with self.assertRaisesRegex(RuntimeError,'quota_exceeded'):
            await asyncio.wait_for(tts.wait_until_spoken(),timeout=0.1)

    async def test_failed_reader_cannot_be_reused_for_a_new_reply(self):
        tts=self.make_tts(Socket())
        await tts._read()
        with self.assertRaisesRegex(RuntimeError,'closed'):tts.start('next')

    async def test_successful_reply_finishes_but_closed_reader_cannot_be_reused(self):
        tts=self.make_tts(Socket([{'contextId':'reply','isFinal':True}]))
        await tts._read()
        await asyncio.wait_for(tts.wait_until_spoken(),timeout=0.1)
        with self.assertRaisesRegex(RuntimeError,'closed'):tts.start('next')

    async def test_send_side_connection_loss_is_not_swallowed(self):
        import websockets
        class ClosedSocket:
            async def send(self,message):raise websockets.ConnectionClosed('lost connection')
        tts=self.make_tts(ClosedSocket())
        with self.assertRaises(websockets.ConnectionClosed):await tts.push('Hello')

    async def test_successful_reconnect_allows_a_new_context(self):
        tts=self.make_tts(Socket())
        await tts._read()
        class OpenSocket:
            def __aiter__(self):
                async def messages():
                    await asyncio.Event().wait()
                    yield
                return messages()
        async def connect(*args,**kwargs):return OpenSocket()
        with patch('elevenlabs_tts.websockets.connect',connect,create=True):await tts.connect()
        tts.start('next')
        self.assertEqual(tts.context,'next')
        tts._reader.cancel()
        await asyncio.gather(tts._reader,return_exceptions=True)
