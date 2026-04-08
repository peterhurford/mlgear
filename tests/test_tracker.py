from mlgear.tracker import Tracker


class TestTracker:
    def test_init(self, capsys):
        t = Tracker()
        captured = capsys.readouterr()
        assert 'Tracker started at' in captured.out

    def test_tick(self, capsys):
        t = Tracker()
        _ = capsys.readouterr()  # clear init output
        t.tick('step 1')
        captured = capsys.readouterr()
        assert 'step 1' in captured.out

    def test_tick_silent(self, capsys):
        t = Tracker()
        _ = capsys.readouterr()
        t.tick('step 1', verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ''
